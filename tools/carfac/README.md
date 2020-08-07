This is a very thin Go wrapper around https://github.com/google/carfac

The libcarfac.a file is checked in, and should enable building, testing
and using this library without any extra manoeuvres.

To recreate the libcarfac.a file from scratch:

- Make sure the git submodule is properly updated: `git submodule update --init`
- Make sure you have installed Eigen3: `sudo apt-get install libeigen3-dev`
- Generate the object file for Go to link against: `go generate`

