package io.github.wlx7.guandan;

/** The table server (android/server, Rust). */
final class Native {
    static {
        System.loadLibrary("guandan_server");
    }

    private Native() {}

    /** Starts the server (once per process); returns the port, or -1 (see lastError). */
    static native int start(String home, String model, int port);

    static native String lastError();

    /** Players and watchers connected to tables right now. */
    static native int connections();

    /** Open tables. */
    static native int tables();

    /** Address for the other devices on the Wi-Fi, or "". */
    static native String lanUrl();
}
