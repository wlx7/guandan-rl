//! JNI entry points for io.github.wlx7.guandan.Native (see android/app).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;

use jni::objects::{JClass, JString};
use jni::sys::{jint, jstring};
use jni::JNIEnv;

static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn set_error(e: String) {
    *LAST_ERROR.lock().unwrap_or_else(|p| p.into_inner()) = e;
}

fn jstr(env: &mut JNIEnv, s: &JString) -> String {
    env.get_string(s).map(|s| s.into()).unwrap_or_default()
}

fn to_java(env: &mut JNIEnv, s: &str) -> jstring {
    env.new_string(s).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_io_github_wlx7_guandan_Native_start(
    mut env: JNIEnv,
    _class: JClass,
    home: JString,
    model: JString,
    port: jint,
) -> jint {
    let home = jstr(&mut env, &home);
    let model = jstr(&mut env, &model);
    let r = catch_unwind(AssertUnwindSafe(|| {
        crate::start(crate::Config {
            home: PathBuf::from(home),
            model: PathBuf::from(model),
            port: port.clamp(1, 65535) as u16,
            platform: "android".into(),
        })
    }));
    match r {
        Ok(Ok(p)) => p as jint,
        Ok(Err(e)) => {
            set_error(e);
            -1
        }
        Err(_) => {
            set_error("服务启动时出错".into());
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_io_github_wlx7_guandan_Native_lastError(mut env: JNIEnv, _class: JClass) -> jstring {
    let e = LAST_ERROR.lock().map(|e| e.clone()).unwrap_or_default();
    to_java(&mut env, &e)
}

#[no_mangle]
pub extern "system" fn Java_io_github_wlx7_guandan_Native_connections(_env: JNIEnv, _class: JClass) -> jint {
    catch_unwind(crate::connections).unwrap_or(0) as jint
}

#[no_mangle]
pub extern "system" fn Java_io_github_wlx7_guandan_Native_tables(_env: JNIEnv, _class: JClass) -> jint {
    catch_unwind(crate::tables).unwrap_or(0) as jint
}

#[no_mangle]
pub extern "system" fn Java_io_github_wlx7_guandan_Native_lanUrl(mut env: JNIEnv, _class: JClass) -> jstring {
    let u = catch_unwind(crate::lan_url).unwrap_or_default();
    to_java(&mut env, &u)
}
