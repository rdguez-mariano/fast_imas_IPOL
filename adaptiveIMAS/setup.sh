#!/bin/bash
mkdir -p imasbuild && cd imasbuild && cmake ../.. && make
mv main z_main
