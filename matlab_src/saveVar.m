function saveVar(var, varName)
    path = '../report_src/variables/';
    f = fopen([path varName '.tex'], 'w+');
    fprintf(f, num2str(var));
    fclose(f);