#!/bin/bash

# compile.sh

mkdir -p bin
javac -cp jgrapht/jgrapht-core/target/classes/:. -d bin */*java