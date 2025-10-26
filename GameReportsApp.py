import streamlit as st
import pandas as pd
import numpy as np
import requests 
import json
import ast
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import plotly.express as px
import plotly.graph_objects as go
from GameReportsCourt import CourtCoordinates
import math
from numpy.random import randint
from nba_api.stats.static import players
from zipfile import ZipFile


st.set_page_config(layout="wide", page_title="NBA Shot Quality Predictor", page_icon="🏀")

def calculate_distance(x1, y1, x2, y2):
    """Calculate the distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_arc_points(p1, p2, apex, num_points=100):
    """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
    t = np.linspace(0, 1, num_points)
    x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
    y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
    z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
    return x, y, z

def create_nba_full_court(ax=None, court_color='#dfbb85',
                          lw=3, lines_color='black', lines_alpha=0.8,
                          paint_fill='', paint_alpha=0.8,
                          inner_arc=False):
    if ax is None:
        ax = plt.gca()

    # Create Pathes for Court Lines
    center_circle = Circle((94/2, 50/2), 6,
                           linewidth=lw, color=lines_color, lw=lw,
                           fill=False, alpha=lines_alpha)
    hoop_left = Circle((5.25, 50/2), 1.5 / 2,
                       linewidth=lw, color=lines_color, lw=lw,
                       fill=False, alpha=lines_alpha)
    hoop_right = Circle((94-5.25, 50/2), 1.5 / 2,
                        linewidth=lw, color=lines_color, lw=lw,
                        fill=False, alpha=lines_alpha)

    # Paint: 16 feet wide, 19 feet from baseline
    left_paint = Rectangle((0, 50/2 - 8), 19, 16,
                           fill=paint_fill, alpha=paint_alpha,
                           lw=lw, edgecolor=None)
    right_paint = Rectangle((94-19, 50/2 - 8), 19, 16,
                            fill=paint_fill, alpha=paint_alpha,
                            lw=lw, edgecolor=None)

    left_paint_border = Rectangle((0, 50/2 - 8), 19, 16,
                                  fill=False, alpha=lines_alpha,
                                  lw=lw, edgecolor=lines_color)
    right_paint_border = Rectangle((94-19, 50/2 - 8), 19, 16,
                                   fill=False, alpha=lines_alpha,
                                   lw=lw, edgecolor=lines_color)

    # Arcs at top of paint (6ft radius)
    left_arc = Arc((19, 50/2), 12, 12, theta1=-90, theta2=90,
                   color=lines_color, lw=lw, alpha=lines_alpha)
    right_arc = Arc((94-19, 50/2), 12, 12, theta1=90, theta2=-90,
                    color=lines_color, lw=lw, alpha=lines_alpha)

    # Lane markers (using standard spacing — could be customized more)
    # You can replicate lane markers symmetrically like before

    # Three-point lines (23.75 ft from basket, 22 ft in corners)
    corner_y = 3  # 22 ft from hoop to corner along baseline
    arc_radius = 23.75
    arc_diameter = arc_radius * 2
    three_pt_left = Arc((5.25, 50/2), arc_diameter, arc_diameter,
                        theta1=-69, theta2=69,
                        color=lines_color, lw=lw, alpha=lines_alpha)
    three_pt_right = Arc((94-5.25, 50/2), arc_diameter, arc_diameter,
                         theta1=180-69, theta2=180+69,
                         color=lines_color, lw=lw, alpha=lines_alpha)

    # 22-foot corner line (~14 feet from baseline to break point)
    ax.plot((0, 14), (corner_y, corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((0, 14), (50-corner_y, 50-corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-14, 94), (corner_y, corner_y), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-14, 94), (50-corner_y, 50-corner_y), color=lines_color, lw=lw, alpha=lines_alpha)

    # Add Patches
    ax.add_patch(left_paint)
    ax.add_patch(right_paint)
    ax.add_patch(left_paint_border)
    ax.add_patch(right_paint_border)
    ax.add_patch(center_circle)
    ax.add_patch(hoop_left)
    ax.add_patch(hoop_right)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)
    ax.add_patch(three_pt_left)
    ax.add_patch(three_pt_right)

    if inner_arc:
        left_inner_arc = Arc((19, 50/2), 12, 12, theta1=90, theta2=-90,
                             color=lines_color, lw=lw, alpha=lines_alpha, ls='--')
        right_inner_arc = Arc((94-19, 50/2), 12, 12, theta1=-90, theta2=90,
                              color=lines_color, lw=lw, alpha=lines_alpha, ls='--')
        ax.add_patch(left_inner_arc)
        ax.add_patch(right_inner_arc)

    # Restricted area (4 ft radius from center of basket)
    restricted_left = Arc((5.25, 50/2), 8, 8, theta1=-90, theta2=90,
                          color=lines_color, lw=lw, alpha=lines_alpha)
    restricted_right = Arc((94-5.25, 50/2), 8, 8, theta1=90, theta2=-90,
                           color=lines_color, lw=lw, alpha=lines_alpha)
    ax.add_patch(restricted_left)
    ax.add_patch(restricted_right)

    # Backboards
    ax.plot((4, 4), ((50/2) - 3, (50/2) + 3), color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((94-4, 94-4), ((50/2) - 3, (50/2) + 3), color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot((4, 4.6), (50/2, 50/2), color=lines_color, lw=lw, alpha=lines_alpha)
    ax.plot((94-4, 94-4.6), (50/2, 50/2), color=lines_color, lw=lw, alpha=lines_alpha)

    # Halfcourt line
    ax.axvline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

    # Court border
    border = Rectangle((0.3, 0.3), 94 - 0.4, 50 - 0.4, fill=False, lw=3,
                       color='black', alpha=lines_alpha)
    ax.add_patch(border)

    # Plot Limit
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.set_facecolor(court_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    return ax


def pts_value(row):
    if (row['made'] == 1):
        if (row['action_type'] == '2pt'):
            return 2
        else:
            return 3
    else:
        return 0

def safe_literal_eval(val):
    if isinstance(val, str) and val.strip().startswith('['):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return val  

def display_player_image(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
        

st.markdown("<h1 style='text-align: center; text-color: #000000; text-shadow: 2px 2px 5px #ffffff; font-size: 5em;'>NBA Shot Quality Predictor</h1>", unsafe_allow_html=True)

selected_competition = st.selectbox("Select Season", ['2021-22','2022-23','2023-24','2024-25'])
zip_path = f'{selected_competition}_top_25_ppg.csv.zip'

with ZipFile(zip_path) as z:
    csv_file = [f for f in z.namelist() if not f.startswith('__MACOSX')][0]
    data = pd.read_csv(z.open(csv_file))
data['pts'] = np.where(data['action_type'] == '2pt',2,3)

playerlist = data['shooter_name'].unique()
selected_player = st.selectbox("Select a player", playerlist)


data = data[(data['shooter_name'] == selected_player) & (data['court_player_name'] == selected_player)]
nba_pid = players.find_players_by_full_name(selected_player)[0]['id']
display_player_image(nba_pid, 350, selected_player)

features = ['minutes',
'seconds',
'spacing',
'Defenders in Paint',
'Shooter in the Paint',
'Closest Defender Behind',
'Closest Defender Inside',
'Closest Defender Blocking',
'Shooter in Restricted Area',
'Second-Closest Defender Behind',
'Second-Closest Defender Inside',
'Harmonic Mean Defender Distance',
'Average Defender Distance (Feet)',
'Defenders in the Restricted Area',
'Second-Closest Defender Blocking',
'Number of Defenders Closer to the Basket',
'Number of Teammates in the Restricted Area',
'Cosine of the Angle with the Closest Defender',
'Cosine of the Angle with the Second-Closest Defender',
"Height of the Closest Defender's Bounding Box (Pixels)",
"Height of the Third-Closest Defender's Bounding Box (Pixels)",
"Height of the Second-Closest Defender's Bounding Box (Pixels)",
'Shot Distance (ft.)',
'Distance to the Closest Defender (Feet)',
'Distance to the Second-Closest Defender (Feet)',
'pts',
'alley_oop',
'catch_and_shoot',
'dunk',
'floater',
'in_paint',
'jumpshot',
'lay_up',
'off_cut',
'off_drive',
'pick_and_roll',
'post_up',
'reverse',
'step_back',
'tip_in',
'transition',
'hook',
'pull_up',
'turnaround']

model = pickle.load(open('xFG_model.pkl', 'rb'))

descriptor_features = ['alley_oop',
'catch_and_shoot',
'dunk',
'floater',
'hook',
'in_paint',
'jumpshot',
'lay_up',
'off_cut',
'off_drive',
'pick_and_roll',
'post_up',
'pull_up',
'reverse',
'step_back',
'tip_in',
'transition',
'turnaround']

data['play_descriptors_parsed'] = data['play_descriptors'].apply(safe_literal_eval)

for desc in descriptor_features:
    data[desc] = data['play_descriptors_parsed'].apply(lambda x: int(desc in x))

input = data[features]
input['spacing'] = input['spacing'].astype(float)
data['pts_value'] = data.apply(pts_value, axis=1)


preds = model.predict_proba(input)[:,1]

data['xFG%'] = preds
data['xpts'] = data['xFG%'] * data['pts']
# defpbp['xFG%'] = preds2
# defpbp['xpts'] = defpbp['xFG%'] * defpbp['pts']
data['player_name'] = data['shooter_name']

st.sidebar.subheader('Filters')

maxshotdist = data['Shot Distance (ft.)'].max()
minshotdist = data['Shot Distance (ft.)'].min()

min_shotdist, max_shotdist = st.sidebar.slider(
    'Shot Distance Range (ft)',
    minshotdist, maxshotdist, (minshotdist, maxshotdist)
)
data = data[(data['Shot Distance (ft.)'] >= min_shotdist) & (data['Shot Distance (ft.)'] <= max_shotdist)]

maxdefdist = data['Distance to the Closest Defender (Feet)'].max()
mindefdist = data['Distance to the Closest Defender (Feet)'].min()
min_defdist, max_defdist = st.sidebar.slider(
    'Defender Distance Range (ft)',
    mindefdist, maxdefdist, (mindefdist, maxdefdist)
)
data = data[(data['Distance to the Closest Defender (Feet)'] >= min_defdist) & (data['Distance to the Closest Defender (Feet)'] <= max_defdist)]

maxXFG = data['xFG%'].max()
minXFG = data['xFG%'].min()

minXFG, maxXFG = st.sidebar.slider(
    'xFG% Range',
    minXFG, maxXFG, (minXFG, maxXFG)
)
data = data[(data['xFG%'] >= minXFG) & (data['xFG%'] <= maxXFG)]

player_xfg = pd.DataFrame(data[data['xFG%'].notna()].groupby('player_name')['xFG%'].mean())
player_xfg['FG%'] = data.groupby('player_name')['made'].mean()
player_xfg['xFG%_OE'] = player_xfg['FG%'] - player_xfg['xFG%']
player_xfg['shots'] = data.groupby('player_name')['made'].count()
player_xfg['3FG%'] = data[data['pts'] == 3].groupby('player_name')['made'].mean()
player_xfg['x3FG%'] = data[data['pts'] == 3].groupby('player_name')['xFG%'].mean()


player_xfg['pps'] = data.groupby('player_name')['pts_value'].sum() / player_xfg['shots']
player_xfg['x_pps'] = data.groupby('player_name')['xpts'].mean()

fg_points = data.groupby('player_name')['pts_value'].sum()

player_xfg['fg_points'] = fg_points
x_fg_points = data.groupby('player_name')['xpts'].sum()
player_xfg['x_fg_points'] = x_fg_points


# player_xfg['x_total_points'] = x_shot_points
player_xfg['avg_shot_dist'] = data[data['xFG%'].notna()].groupby('player_name')['Shot Distance (ft.)'].mean()
player_xfg['avg_def_height'] = data[data['xFG%'].notna()].groupby('player_name')['Height of the Closest Defender (Inches)'].mean()
player_xfg['avg_def_weight'] = data[data['xFG%'].notna()].groupby('player_name')['Weight of the Closest Defender (Pounds)'].mean()
player_xfg['avg_def_dist'] = data[data['xFG%'].notna()].groupby('player_name')['Distance to the Closest Defender (Feet)'].mean()



if not player_xfg.empty:
    st.markdown('---')
    colA, colB = st.columns(2)
    if not player_xfg.empty:
        pxfg = player_xfg[player_xfg['shots'] > 5].sort_values(by='xFG%_OE', ascending=False).reset_index()
        pxfg = pxfg.iloc[0] if not pxfg.empty else None
        with colA:
            st.markdown('### Offensive Shot Summary')
            if pxfg is not None:
                m1, m2,m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div style='font-size:18px; margin-bottom:8px;'><b>FG%:</b> {pxfg['FG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG%:</b> {pxfg['xFG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>3FG%:</b> {pxfg['3FG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>x3FG%:</b> {pxfg['x3FG%']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xFG% OE:</b> {pxfg['xFG%_OE']:.4f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Shots:</b> {int(pxfg['shots'])}</div>

                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div style='font-size:18px; margin-bottom:8px;'><b>PPS:</b> {pxfg['pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>xPPS:</b> {pxfg['x_pps']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Shot Distance (ft):</b> {pxfg['avg_shot_dist']:.2f}</div>
                        """, unsafe_allow_html=True)
                with m3:
                     st.markdown(f"""
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Distance from Defender (ft):</b> {pxfg['avg_def_dist']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Defender Height (in):</b> {pxfg['avg_def_height']:.2f}</div>
                    <div style='font-size:18px; margin-bottom:8px;'><b>Avg Defender Weight (lbs):</b> {pxfg['avg_def_weight']:.2f}</div>
                        """, unsafe_allow_html=True)
    st.markdown('---')

xFG = st.checkbox('xFG%',value=True)
sameSide = st.checkbox('Same Side of Court')
if sameSide:
    data.loc[data['shot_x'] < 47, 'shot_x'] = 94 - data['shot_x']
if xFG:
    courtbg = 'black'
    courtlines = 'white'
else:
    courtbg = '#d2a679'
    courtlines = 'black'
import plotly.express as px
court = CourtCoordinates()
court_lines_df = court.get_court_lines()

fig = px.line_3d(
    data_frame=court_lines_df,
    x='x',
    y='y',
    z='z',
    line_group='line_group',
    color='color',
    color_discrete_map={
        'court': courtlines,
        'hoop': '#e47041',
        'net': '#D3D3D3',
        'backboard': 'gray'
    }
)
fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)

fig.update_traces(line=dict(width=5))
fig.update_layout(    
    margin=dict(l=20, r=20, t=20, b=20),
    scene_aspectmode="data",
    height=600,
    scene_camera=dict(
        eye=dict(x=1.3, y=0, z=0.7)
    ),
    scene=dict(
        xaxis=dict(title='', showticklabels=False, showgrid=False),
        yaxis=dict(title='', showticklabels=False, showgrid=False),
        zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=True, backgroundcolor=courtbg),
    ),
    showlegend=False,
    legend=dict(
        yanchor='top',
        y=0.05,
        x=0.2,
        xanchor='left',
        orientation='h',
        font=dict(size=15, color='gray'),
        bgcolor='rgba(0, 0, 0, 0)',
        title='',
        itemsizing='constant'
    )
)
if xFG:
    fig.add_trace(go.Scatter3d(
        x=data['shot_y'], y=data['shot_x'], z=[0]*len(data),
        mode='markers+text',
        marker=dict(size=7, color=data['xFG%'], colorscale='Hot', cmin=0, cmax=1, opacity=0.8),
        hovertemplate=
        'Shooter: %{customdata[4]}<br>' +
        'Shot Distance: %{customdata[0]:.1f} ft<br>' +
        'Defender Distance: %{customdata[1]:.1f} ft<br>' +
        'Shot Result: %{customdata[2]}<br>' +
        'Shot Type: %{customdata[3]}<br>' +
        'Shot Descriptors: %{customdata[5]}<br>' +
        'xFG%: %{customdata[7]:.2f}<br>' +
        'xPts: %{customdata[6]:.2f}<br>',

    customdata=np.stack((
        data['Shot Distance (ft.)'],
        data['Distance to the Closest Defender (Feet)'],
        np.where(data['made'] == 1, 'Made', 'Missed'),
        data['action_type'],
        data['player_name'],
        data['play_descriptors'],
        data['xpts'],
        data['xFG%']
    ), axis=-1)
    ))
else:
    fig.add_trace(go.Scatter3d(
        x=data['shot_y'], y=data['shot_x'], z=[0]*len(data),
        mode='markers+text',
        marker=dict(size=7, color=np.where(data['made'] == 1, 'green', 'red'), opacity=0.8),
        hovertemplate=
        'Shooter: %{customdata[4]}<br>' +
        'Shot Distance: %{customdata[0]:.1f} ft<br>' +
        'Defender Distance: %{customdata[1]:.1f} ft<br>' +
        'Shot Result: %{customdata[2]}<br>' +
        'Shot Type: %{customdata[3]}<br>' +
        'Shot Descriptors: %{customdata[5]}<br>' +
        'xFG%: %{customdata[7]:.2f}<br>' +
        'xPts: %{customdata[6]:.2f}<br>',

    customdata=np.stack((
        data['Shot Distance (ft.)'],
        data['Distance to the Closest Defender (Feet)'],
        np.where(data['made'] == 1, 'Made', 'Missed'),
        data['action_type'],
        data['player_name'],
        data['play_descriptors'],
        data['xpts'],
        data['xFG%']
    ), axis=-1)
    ))

x_values = []
y_values = []
z_values = []
x_values2 = []
y_values2 = []
z_values2 = []
offpbp_made = data[data['made'] == 1]

for index, row in offpbp_made.iterrows():
    x_values.append(row['shot_y'])
    y_values.append(row['shot_x'])
    z_values.append(0)
    x_values2.append(court.hoop_loc_x)
    if row['shot_x'] <= 47:
        y_values2.append(court.hoop_loc_y)
    else:
        y_values2.append(court.court_length-court.hoop_loc_y)
    z_values2.append(court.hoop_loc_z)
x_coords = x_values
y_coords = y_values
z_value = 0  # Fixed z value
x_coords2 = x_values2
y_coords2 = y_values2
z_value2 = court.hoop_loc_z

for i in range(len(offpbp_made)):
    x1 = x_coords[i]
    y1 = y_coords[i]
    x2 = x_coords2[i]
    y2 = y_coords2[i]
    p2 = np.array([x1, y1, z_value])
    p1 = np.array([x2, y2, z_value2])
    distance = calculate_distance(x1, y1, x2, y2)
    # Set arc height based on shot distance
    shot_dist = offpbp_made['Shot Distance (ft.)'].iloc[i]
    if shot_dist > 3:
        if shot_dist > 50:
            h = randint(255,305)/10
        elif shot_dist > 30:
            h = randint(230,280)/10
        elif shot_dist > 25:
            h = randint(180,230)/10
        elif shot_dist > 15:
            h = randint(180,230)/10
        else:
            h = randint(130,160)/10
        apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
        x, y, z = generate_arc_points(p1, p2, apex)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=8, color='green'),
            opacity=0.5,
            hoverinfo='none',
        ))

st.plotly_chart(fig, use_container_width=True)

# Organize offensive plots in columns
if not data.empty:
        col1, col2, col3 = st.columns(3)
        # 1. Line chart: xFG% and FG% by game date
        with col1:
            if selected_competition != '2024-25':
                data['game_datetime_utc'] = pd.to_datetime(data['game_datetime_utc'])
            else:
                data['game_datetime_utc'] = data['game_id_x']
            by_game = data.groupby('game_datetime_utc').agg({
                'xFG%': 'mean',
                'made': 'mean',
                'pts_value': 'count'
            }).reset_index().sort_values('game_datetime_utc')
            by_game['FG%'] = by_game['made']
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=by_game['game_datetime_utc'], y=by_game['xFG%'], mode='lines+markers', name='xFG%', line=dict(color='orange')))
            fig1.add_trace(go.Scatter(x=by_game['game_datetime_utc'], y=by_game['FG%'], mode='lines+markers', name='FG%', line=dict(color='green')))
            fig1.update_layout(title='xFG% and FG% by Game', xaxis_title='Game Date', yaxis_title='Percentage', legend_title='Metric')
            st.plotly_chart(fig1, use_container_width=True)

        with col3:
            data['game_datetime_utc'] = pd.to_datetime(data['game_datetime_utc'])
            by_game = data.groupby('game_datetime_utc').agg({
                'xpts': 'sum',
                'pts_value': 'sum',
                'pts_value': 'count'
            }).reset_index().sort_values('game_datetime_utc')
            # by_game['FG%'] = by_game['made']
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=by_game['game_datetime_utc'], y=by_game['xpts'], mode='lines+markers', name='xpts', line=dict(color='orange')))
            fig1.add_trace(go.Scatter(x=by_game['game_datetime_utc'], y=by_game['pts_value'], mode='lines+markers', name='pts_value', line=dict(color='green')))
            fig1.update_layout(title='Expected and Actual Field Goal PPG', xaxis_title='Game Date', yaxis_title='Points', legend_title='Metric')
            st.plotly_chart(fig1, use_container_width=True)
        # 2. Bar chart: Shot type frequency
        with col2:
            shot_types = data['action_type'].value_counts().reset_index()
            shot_types.columns = ['Shot Type', 'Count']
            fig2 = px.bar(shot_types, x='Shot Type', y='Count', color='Count', color_continuous_scale='Blues', title='Shot Type Frequency')
            st.plotly_chart(fig2, use_container_width=True)
        col4, col5, col6 = st.columns(3)
        # 3. Scatter plot: Shot distance vs. xFG%, colored by make/miss
        with col4:
            fig3 = px.scatter(
                data, x='Shot Distance (ft.)', y='xFG%', color=data['made'].map({1: 'Made', 0: 'Missed'}),
                hover_data=['game_datetime_utc', 'action_type', 'play_descriptors'],
                title='Shot Distance vs. xFG% (Colored by Result)',
                color_discrete_map={'Made': 'green', 'Missed': 'red'}
            )
            st.plotly_chart(fig3, use_container_width=True)
        # 4. Histogram: Distribution of shot distances
        with col6:
            fig4 = px.histogram(data, x='Shot Distance (ft.)', nbins=20, color_discrete_sequence=['#636EFA'])
            fig4.update_layout(title='Distribution of Shot Distances', xaxis_title='Shot Distance (ft.)', yaxis_title='Count')
            st.plotly_chart(fig4, use_container_width=True)
        # 5. Radar chart: Descriptor features profile
        with col5:
            desc_counts = {desc: data[desc].sum() for desc in descriptor_features}
            radar_df = pd.DataFrame({'Descriptor': list(desc_counts.keys()), 'Count': list(desc_counts.values())})
            fig5 = go.Figure()
            fig5.add_trace(go.Scatterpolar(
                r=radar_df['Count'],
                theta=radar_df['Descriptor'],
                fill='toself',
                name='Descriptor Profile',
                line=dict(color='green')
            ))
            fig5.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title='Shot Descriptor Profile (Radar Chart)'
            )
            st.plotly_chart(fig5, use_container_width=True)





