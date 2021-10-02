import numpy as np
import plotly.express as px
import dash

# for data import
import os
from glob import glob
import lz4.frame

import gzip
import nibabel as nib
import shutil

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Plotting
# For reference: mri dims: dim0=saggital, dim1=coronal, dim2=axial
import chart_studio.plotly as py
import plotly.graph_objects as go
from datetime import date

# import test brain volume
def load_image(path: str):
    if path.endswith('.npy.lz4'):
        with lz4.frame.open(path, 'rb') as f:
            return np.load(f)
    elif path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nii.gz'):
        return nib.load(path).get_fdata() # open as numpy array
    else:
        raise Exception("File extension not supported!")

def main():

    TEST_VOL = '/usr/local/data/kvasilev/mais/data/BraTS19_CBICA_ASK_1_t2.nii.gz'
    stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    # load data
    data = load_image(TEST_VOL)

    # using dash
    app = dash.Dash(__name__, external_stylesheets=stylesheet)

    layout = [
            html.P('MRI Visualizer'),
            dcc.Graph(id='mri-plot',style={'width': '90vh', 'height': '90vh'}),
            dcc.Slider(id='slice-slider', min=0,max=88,step=1,value=40, marks={0:'0', 88:'88'}),
            dcc.Slider(id='age-slider', min=60, max=80, marks={60: 'Age 60', 80:'Age 80'}, value=60),
            dcc.RadioItems(id='slice-selector',
                            options=[{'label':'Axial', 'value': 'axial'},
                                    {'label':'Coronal', 'value':'coronal'},
                                    {'label':'Saggital', 'value':'saggital'}],
                                    value='axial',
                                    labelStyle={'display':'inline-block'}),
            
            dcc.DatePickerSingle(id='bdate-picker',
                            date=date(1969,4,20)),
            html.Button('Submit', id='button')]

    app.layout = html.Div(layout)

    @app.callback(
        Output('mri-plot', 'figure'),
        [Input('slice-slider','value'),
        Input('age-slider','value'),
        Input('slice-selector', 'value'),
        Input('bdate-picker','date'),
        ])    # ADD BUTTON BACK
    def visualize_mri(slice_num, age_num, slice_type, bdate_num):

        # plot appropriate slice
        if slice_type=='saggital': #dim0
            fig = px.imshow(data[slice_num,:,:])
        elif slice_type=='coronal': #dim1
            fig = px.imshow(data[:,slice_num,:])
        elif slice_type=='axial': #dim2
            fig = px.imshow(data[:,:,slice_num])
        else:
            print('Invalid axis selection!')
        
        return fig

    # show         
    app.run_server(port=42000,debug=True)

if __name__== '__main__':
    main()