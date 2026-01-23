@echo off
set lib=../png;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/lib/x64;C:/Program Files (x86)/Windows Kits/NETFXSDK/4.8/lib/um/x64;C:/Program Files (x86)/Windows Kits/10/lib/10.0.22621.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/lib/10.0.22621.0/um/x64
set include=../;./;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/include;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/VS/include;C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/ucrt;C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/um;C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/shared;C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/winrt;C:/Program Files (x86)/Windows Kits/10/include/10.0.22621.0/cppwinrt;C:/Program Files (x86)/Windows Kits/NETFXSDK/4.8/include/um

cl maingl.c /c /nologo /O2 /fp:fast /arch:AVX2
link maingl.obj /out:nn.exe libpng16.lib opengl32.lib user32.lib glu32.lib gdi32.lib /nologo
erase maingl.obj