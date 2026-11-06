package io.github.wlx7.guandan;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;

/**
 * Keeps the table server running while the app is in the background, so that the other phones
 * at a table hosted here can keep playing. Shows the LAN address in a notification with a
 * "退出" button, and stops by itself after 10 minutes in the background with nobody at a table.
 */
public class ServerService extends Service {
    private static final String CHANNEL = "server";
    private static final String ACTION_QUIT = "io.github.wlx7.guandan.QUIT";
    private static final int ID = 1;
    private static final long TICK_MS = 30_000;
    private static final long IDLE_LIMIT_MS = 10 * 60_000;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private PowerManager.WakeLock cpu;
    private WifiManager.WifiLock wifi;
    private long idleMs = 0;
    private String shown = "";

    static void start(Context ctx) {
        Intent i = new Intent(ctx, ServerService.class);
        try {
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(i);
            else ctx.startService(i);
        } catch (Exception ignored) {
            // e.g. not allowed from the background; the activity starts it again on resume
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        channel();
        goForeground(text());
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        if (pm != null) {
            cpu = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "guandan:server");
            cpu.setReferenceCounted(false);
            cpu.acquire();
        }
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(WIFI_SERVICE);
        if (wm != null) {
            //noinspection deprecation
            wifi = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "guandan:server");
            wifi.setReferenceCounted(false);
            wifi.acquire();
        }
        handler.postDelayed(this::tick, TICK_MS);
    }

    /** Creates the notification channel, or renames it after the language changed. */
    private void channel() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= 26 && nm != null) {
            NotificationChannel ch = new NotificationChannel(CHANNEL, Lang.pick(this, "牌桌服务", "Table server"), NotificationManager.IMPORTANCE_LOW);
            ch.setDescription(Lang.pick(this, "局域网牌桌运行时显示，可以从这里退出", "Shown while LAN tables run; you can quit from here"));
            ch.setShowBadge(false);
            nm.createNotificationChannel(ch);
        }
    }

    private String text() {
        String url = "";
        int tables = 0;
        try {
            url = Native.lanUrl();
            tables = Native.tables();
        } catch (Throwable ignored) {
        }
        if (Lang.en(this)) {
            String where = url.isEmpty() ? "Not on WiFi: single-device play only" : "Phones on the same WiFi open " + url;
            return (tables > 0 ? tables + (tables == 1 ? " table playing · " : " tables playing · ") : "") + where;
        }
        String where = url.isEmpty() ? "没有连上 WiFi，只能单机打" : "同一 WiFi 的手机打开 " + url;
        return (tables > 0 ? tables + " 桌进行中 · " : "") + where;
    }

    private void goForeground(String text) {
        Notification n = build(text);
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(ID, n);
        }
        shown = text;
    }

    @SuppressWarnings("deprecation")
    private Notification build(String text) {
        int flags = PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= 23 ? PendingIntent.FLAG_IMMUTABLE : 0);
        Intent open = new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent openPi = PendingIntent.getActivity(this, 0, open, flags);
        PendingIntent quitPi = PendingIntent.getService(this, 1, new Intent(this, ServerService.class).setAction(ACTION_QUIT), flags);
        Notification.Builder b = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this, CHANNEL) : new Notification.Builder(this);
        b.setSmallIcon(R.drawable.ic_stat)
                .setContentTitle(Lang.pick(this, "掼蛋 AI 正在运行", "Guandan AI is running"))
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setContentIntent(openPi)
                .setOngoing(true)
                .setShowWhen(false)
                .addAction(new Notification.Action.Builder(null, Lang.pick(this, "退出", "Quit"), quitPi).build());
        return b.build();
    }

    private void tick() {
        int conns = 0;
        try {
            conns = Native.connections();
        } catch (Throwable ignored) {
        }
        idleMs = (MainActivity.visible || conns > 0) ? 0 : idleMs + TICK_MS;
        if (idleMs >= IDLE_LIMIT_MS) {
            quit();
            return;
        }
        String t = text();
        if (!t.equals(shown)) {
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) nm.notify(ID, build(t));
            shown = t;
        }
        handler.postDelayed(this::tick, TICK_MS);
    }

    private void quit() {
        handler.removeCallbacksAndMessages(null);
        stopForeground(true);
        stopSelf();
        // the server runs in this process; ending the process ends every table
        handler.postDelayed(() -> {
            android.os.Process.killProcess(android.os.Process.myPid());
        }, 300);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_QUIT.equals(intent.getAction())) {
            quit();
            return START_NOT_STICKY;
        }
        channel();
        goForeground(text()); // required after every startForegroundService
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        handler.removeCallbacksAndMessages(null);
        if (cpu != null && cpu.isHeld()) cpu.release();
        if (wifi != null && wifi.isHeld()) wifi.release();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
