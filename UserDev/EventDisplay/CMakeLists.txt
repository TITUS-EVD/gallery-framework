add_subdirectory(Reader)
if (TITUS_LEGACY)
    add_subdirectory(3DViewer)
endif()

# The python side is not installed here: env.sh pip installs titus
# (editable) from python/ and puts python/ (evd.py) on PATH.
