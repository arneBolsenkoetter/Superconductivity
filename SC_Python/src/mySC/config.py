from pathlib import Path


config_path = Path(__file__).resolve()  # config needs to be located like: /path/to/PROJECT_ROOT/PY_DIR/own_code/config.py

PROJECT_ROOT = config_path.parents[2]
PY_DIR = PROJECT_ROOT / Path('SC_Python')
DATA_DIR = PROJECT_ROOT / Path('SC_Data')
LATEX_DIR = PROJECT_ROOT / Path('SC_Latex')

figrect_norm=(3.375,0.75*3.375) # numbers in inches

def figrect(m:int=1,n:int=1,sw:float=1.0,sh:float=1.0) -> tuple[float,float]:
    """
        Returns a tuple with the size (width,height) for a matplotlib.pyplot (plt) subplots-figure.

        Parameters:
        - m:    int,    multiplies the figure-widtch with 'ncols' = number of columns;
        - n:    int,    multiplies the figure-height with 'nrows' = number of rows;
        - sw:   float,  abbreviation for 'Sqeeze Width'. Stretches of thwats the figure-width by the set value;
        - sh:   float,  abbreviation for 'Squeeze Height', Stretches or thwats the figure_height by set value;

        Return:
        - (fig-width, fig-height):  tuple[float,float], size for a scientific-review-paper-styled image, in INCHES!
    """
    return (3.375*m*sw,0.75*3.375*n*sh)