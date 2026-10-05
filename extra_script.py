Import("env")

# List C++ standard flags to strip out
unflags = [
    "-std=gnu++11",
    "-std=gnu++14",
    "-std=c++11",
    "-std=c++14"
]

# Collect active SCons construction environments
envs = [env, DefaultEnvironment()]
try:
    Import("projenv")
    envs.append(projenv)
except Exception:
    pass

for e in envs:
    # 1. Strip older C++ standard flags from global build env
    for flag in unflags:
        e.ProcessUnFlags(flag)
    
    # 2. Append C++17 standard flag if not already present
    if "-std=gnu++17" not in e.get("CXXFLAGS", []):
        e.Append(CXXFLAGS=["-std=gnu++17"])

print("[tflite-micro] set C++17 compiler")