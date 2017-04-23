def write_variable_to_latex(var, var_name):
    filename = '../report_src/variables/{}.tex'.format(var_name)
    target = open(filename, 'w')
    var = str(var)
    if len(var)>1:
        var = var.replace('  ', ', ')

    target.write(var)
    target.close()