@echo on
echo Building Conda environment 'ai4cyber_project'...
call conda env create -f setup\conda\build\environment.yaml
call conda activate ai4cyber_project
pause