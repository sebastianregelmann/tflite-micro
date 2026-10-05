Import("env")

# Obtain the global construction environment
global_env = DefaultEnvironment()

unflags = [
    "-std=gnu++11",
    "-std=gnu++14",
    "-std=c++11",
    "-std=c++14"
]

# Apply to local library env, global env, and project env
envs = [env, global_env]
try:
    Import("projenv")
    envs.append(projenv)
except Exception:
    pass

for e in envs:
    for flag in unflags:
        e.ProcessUnFlags(flag)
    if "-std=gnu++17" not in e.get("CXXFLAGS", []):
        e.Append(CXXFLAGS=["-std=gnu++17"])

print("[tflite-micro] set C++17 compiler")