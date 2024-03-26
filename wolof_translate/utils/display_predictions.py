import plotly.graph_objects as go
from tabulate import tabulate
import plotly.io as pio
import pandas as pd
import textwrap

def display_samples(data_frame: pd.DataFrame, 
                    n_samples: int = 40, 
                    seed: int = 0, 
                    header_color: str = 'paleturquoise',
                    cells_color: str = 'lavender', width: int = 600, height: int = 1000, 
                    save_sample: bool = True, 
                    table_caption: str = '',
                    label: str = '',
                    filename: str = 'samples.csv', lang: str = 'eng', show: bool = True):
    """Display a random sample of the data frame.

    Args:
        data_frame (pd.DataFrame): The data frame to display.
        n_samples (int, optional): The number of samples. Defaults to 40.
        seed (int, optional): The generator' seed. Defaults to 0.
        header_color (str, optional): The header color. Defaults to 'paleturquoise'.
        cells_color (str, optional): The cells' color. Defaults to 'lavender'.
        width (int): The width of the figure. Defaults to 600.
        height (int): The height of the figure. Defaults to 300.
        lang (str): The language: 'fr' for french or 'eng' for english. Defaults to 'eng'.

    Returns:
        : The figure.
    """
    
    # get the samples from the data frame
    # samples = data_frame.sample(n_samples, random_state = seed).tail(13)
    samples = data_frame.sample(n_samples, random_state = seed)
    
    if lang == 'fr':
        
        samples.columns = ['Phrases Originales', 'Target Sentences', 'Prédictions']
    
    elif lang == 'eng':
        
        samples.columns = ['Source Sentences', 'Target Sentences', 'Predictions']
    
    # trace the figure
    fig = go.Figure(
        data = go.Table(
            header = dict(
                values = list(samples.columns),
                fill_color = header_color,
                align = 'center',
                font=dict(size=14, color='black'),  # Header font style
                height=40
            ),
            cells = dict(
                values = [samples[col] for col in samples.columns],
                fill_color = cells_color,
                line=dict(color='rgb(204, 204, 204)', width=1),  
                font=dict(size=12, color='black'), 
                height=30, 
                align = 'left'
            ),
            columnwidth=[400, 400, 400]
        )
    )
    
    # Customize the table layout
    fig.update_layout(
        width=width,  # Set the overall table width
        height=height,  # Set the overall table height
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margin
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area background
    )
    
    # display the figure
    if show: fig.show()
    
    # save the latex script to create the table in latex
    if save_sample: samples.to_csv(f'{filename}_{lang}.csv', index = False, encoding = 'utf-16')
    
    return fig

def save_go_figure_as_image(fig, path: str, scale: int = 3, width: int = 600, height: int = 1000):
    
    # save the figure as a image
    pio.write_image(fig, path, format='png', scale=scale, width=width, height=height)
    
def escape_latex(text):
    """
    Escape special characters in text for LaTeX.
    """
    special_chars = ['_', '&', '%', '$', '#', '{', '}', '~', '^']
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    return text

def wrap_text(text, max_width):
    """
    Wrap long text to fit into the table cells.
    """
    return '\n'.join(textwrap.wrap(text, width=max_width))

def save_latex_table(data_frame, table_caption='', label='', filename='table.tex',
                     max_cell_width: int = 100, wrap_long_text: bool = True):
    """
    Convert a pandas DataFrame to a LaTeX table and save it to a file.

    Parameters:
        data_frame (pandas.DataFrame): The DataFrame to convert to LaTeX table.
        table_caption (str): Optional caption for the table.
        label (str): Optional label for referencing the table in the document.
        filename (str): The name of the file to save the LaTeX code.

    Returns:
        None
    """
    # Convert the DataFrame to a LaTeX tabular representation
    latex_table = data_frame.to_latex(index=False, escape=False, column_format='p{.425\linewidth}p{.425\linewidth}')

    # Modify the LaTeX tabular representation to include the caption and label, and add necessary formatting
    latex_table = (
        "\\begin{table}\n"
        "  \\centering\n"
        "  \\small\n"
        f"  \\caption{{{table_caption}}}\n\n"
        "  \\begin{tabular}{*{3}{p{.425\\linewidth}}}\n"
        "    \\toprule\n"
        "    Emociones primarias &  Derivación de las emociones primarias \\\\\n"
        "    \\midrule\n"
        f"{latex_table}"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        f"  \\label{{{label}}}\n"
        "\\end{table}"
    )

    with open(filename, 'w', encoding='utf-8') as f:
        
        f.write(latex_table)


