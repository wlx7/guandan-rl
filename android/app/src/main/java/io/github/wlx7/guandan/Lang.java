package io.github.wlx7.guandan;

import android.content.Context;

/**
 * The interface language chosen in the page (中文 by default, or English), remembered for the
 * texts the app itself shows: the notification and the start-up error.
 */
final class Lang {
    private static final String PREFS = "guandan";
    private static final String KEY = "lang";

    private Lang() {
    }

    static boolean en(Context ctx) {
        return "en".equals(ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "zh"));
    }

    /** Returns whether the language changed. */
    static boolean set(Context ctx, String lang) {
        String v = "en".equals(lang) ? "en" : "zh";
        if (v.equals(ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "zh"))) return false;
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, v).apply();
        return true;
    }

    static String pick(Context ctx, String zh, String en) {
        return en(ctx) ? en : zh;
    }
}
