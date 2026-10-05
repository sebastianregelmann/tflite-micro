Import("env")
import os

# Strip default C++ standard flags (-std=c++11, -std=gnu++14, etc.) from project env
for flag_key in ["CXXFLAGS", "BASECXXFLAGS"]:
    if flag_key in env:
        env[flag_key] = [f for f in env[flag_key] if not f.startswith("-std=")]

# Apply C++17 standard globally across the entire project build
env.Append(CXXFLAGS=["-std=gnu++17"])