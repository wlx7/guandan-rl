package io.github.wlx7.guandan;

import android.content.Context;
import android.content.pm.PackageInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Starts the table server: the model file is copied out of the APK once per version. */
final class Server {
    static final int PORT = 8000;
    private static int port = -1;
    private static String error = "";

    private Server() {}

    static synchronized int ensureStarted(Context ctx) {
        if (port > 0) return port;
        try {
            File model = extractModel(ctx);
            int p = Native.start(ctx.getFilesDir().getAbsolutePath(), model.getAbsolutePath(), PORT);
            if (p <= 0) {
                error = Native.lastError();
                return -1;
            }
            port = p;
            return p;
        } catch (Throwable t) {
            error = String.valueOf(t);
            return -1;
        }
    }

    static String error() {
        return error;
    }

    static int port() {
        return port;
    }

    private static File extractModel(Context ctx) throws IOException {
        long code;
        try {
            PackageInfo pi = ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0);
            code = pi.lastUpdateTime;
        } catch (Exception e) {
            code = 0;
        }
        File dir = new File(ctx.getFilesDir(), "model");
        File out = new File(dir, "guandan-medium-" + code + ".gdm");
        if (out.exists() && out.length() > 0) return out;
        //noinspection ResultOfMethodCallIgnored
        dir.mkdirs();
        File[] old = dir.listFiles();
        if (old != null) for (File f : old) //noinspection ResultOfMethodCallIgnored
            f.delete();
        File tmp = new File(dir, "tmp");
        try (InputStream in = ctx.getAssets().open("guandan-medium.gdm");
             OutputStream os = new FileOutputStream(tmp)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        }
        if (!tmp.renameTo(out)) throw new IOException("cannot write " + out);
        return out;
    }
}
