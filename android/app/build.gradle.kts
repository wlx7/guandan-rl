// The Android app: a WebView on the table server (android/server, Rust, in jniLibs) that also
// serves the other devices on the Wi-Fi. Version = the project's version in the root Cargo.toml.
plugins {
    id("com.android.application")
}

val projectVersion: String = Regex("^version = \"(.+)\"", RegexOption.MULTILINE)
    .find(rootDir.resolve("../Cargo.toml").readText())!!.groupValues[1]
val versionParts = projectVersion.split(".").map { it.takeWhile(Char::isDigit).toIntOrNull() ?: 0 }

android {
    namespace = "io.github.wlx7.guandan"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.github.wlx7.guandan"
        minSdk = 24
        targetSdk = 34
        versionName = projectVersion
        versionCode = versionParts[0] * 10000 + versionParts.getOrElse(1) { 0 } * 100 + versionParts.getOrElse(2) { 0 }
    }

    signingConfigs {
        // a fixed test key (public, in the repo) so that every build installs over the previous
        // one; release builds use the key from the CI secrets when they are set
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
        create("release") {
            // CI passes the release key; without it (local builds) the debug key is used
            val ks = System.getenv("ANDROID_KEYSTORE")
            if (ks != null && file(ks).exists()) {
                storeFile = file(ks)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("ANDROID_KEY_ALIAS") ?: "guandan"
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD") ?: System.getenv("ANDROID_KEYSTORE_PASSWORD")
            }
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            val rel = signingConfigs.getByName("release")
            signingConfig = if (rel.storeFile != null) rel else signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}
