#!/bin/bash
mkdir -p imasbuild && cd imasbuild && cmake ../.. && make
mv main ../imas_bin
