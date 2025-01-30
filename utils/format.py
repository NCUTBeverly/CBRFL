import pandas as pd
import pyfiglet


def print_banner(text):
    f = pyfiglet.Figlet(font="slant")
    print(f.renderText(text))


def print_df(df: pd.DataFrame):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None):
        print(df)

def df2str(df: pd.DataFrame):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None):
        return df.to_string()



