#!/bin/bash
echo "Installing/updating conda environment and dependencies"
conda env create -f environment.yml || conda env update
echo "creating .env for autoenv, follow README.md for its installation"
rm .env
echo "source activate compsumm > /dev/null 2>&1" >> .env
echo "Installation completed !!!"
