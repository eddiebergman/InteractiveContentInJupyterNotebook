
from IPython.display import Javascript
from IPython.core.display import HTML

def code_horizontal(idxs):
    for idx in idxs:
        tag_cell([idx], 'horizontal_code_cell')
        tag_code_input(idx, 'half_width')
        tag_code_output(idx, 'half_width')

def code_horizontal_reverse(idxs):
    for idx in idxs:
        tag_cell([idx], 'horizontal_code_cell_reverse')
        tag_code_input(idx, 'half_width')
        tag_code_output(idx, 'half_width')

def get_cell_JSstr(idx):
    return '''
        const notebook_top = $("#notebook-container");
        const element = notebook_top.children()[{}];
    '''.format(idx)

def tag_cell(idxs, tag, identifier='class'):
    for idx in idxs:
        cell_str = get_cell_JSstr(idx)
        if identifier is 'class':
            cell_str = cell_str + '''
                element.className += " {}";
            '''.format(tag)
        elif identifier is 'id':
            cell_str = cell_str + '''
                element.id = "{}";
            '''.format(tag)
        display(Javascript(cell_str))

def tag_code_input(idx, tag, identifier='class'):
    cell_str = get_cell_JSstr(idx)
    if identifier is 'class':
        cell_str = cell_str + '''
            element.children[0].className += " {}";
        '''.format(tag)
    elif identifier is 'id':
        cell_str = cell_str + '''
            element.children[0].id = "{}";
        '''.format(tag)
    display(Javascript(cell_str))


def tag_code_output(idx, tag, identifier='class'):
    cell_str = get_cell_JSstr(idx)
    if identifier is 'class':
        cell_str = cell_str + '''
            element.children[1].className += " {}";
        '''.format(tag)
    elif identifier is 'id':
        cell_str = cell_str + '''
            element.children[1].id = "{}";
        '''.format(tag)
    display(Javascript(cell_str))

def tag_markdown_cell(idx, tag, identifier='class'):
    cell_str = get_cell_JSstr(idx)
    if identifier is 'class':
        cell_str = cell_str + '''
            element.className += " {}";
        '''.format(tag)
    elif identifier is 'id':
        cell_str = cell_str + '''
            element.id = "{}";
        '''.format(tag)
    display(Javascript(cell_str))

def set_css_style(css_file_path):
    """
    Read the custom CSS file and load it into Jupyter.
    Pass the file path to the CSS file.
    """

    styles = open(css_file_path, "r").read()
    return HTML('<style>{}</style>'.format(styles))
