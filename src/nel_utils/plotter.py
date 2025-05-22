import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_evolution_plot(df_log, plot_title, logarithmic=False, var='rmse'):
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    df_plot = pd.DataFrame({
        'x': df_log.iloc[:, 4],
        'rmse': df_log.iloc[:, 5],
        'rmse_val': df_log.iloc[:, 8],
        'size': df_log.iloc[:, 9]
    })

    if var == 'size' and logarithmic:
        # Convert to numeric, coercing errors to NaN
        df_plot['size'] = pd.to_numeric(df_plot['size'], errors='coerce').fillna(0)
        # Now apply log10 safely
        df_plot['size'] = df_plot['size'].apply(lambda v: np.log10(v + 1) if v >= 0 else 0)

    agg = df_plot.groupby('x')[var].agg(['mean', 'std']).reset_index()
    agg['y_upper'] = agg['mean'] + agg['std']
    agg['y_lower'] = agg['mean'] - agg['std']
    agg.loc[agg['y_lower'] < 0, 'y_lower'] = 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg['x'], y=agg['mean'], mode='lines',
        name='Train' if var == 'rmse' else 'Size', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=agg['x'], y=agg['y_upper'], mode='lines',
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=agg['x'], y=agg['y_lower'], mode='lines',
        fill='tonexty', fillcolor='rgba(0,0,255,0.1)',
        line=dict(width=0), showlegend=False
    ))

    if var == 'rmse':
        agg_val = df_plot.groupby('x')['rmse_val'].agg(['mean', 'std']).reset_index()
        agg_val['y_upper'] = agg_val['mean'] + agg_val['std']
        agg_val['y_lower'] = agg_val['mean'] - agg_val['std']

        fig.add_trace(go.Scatter(
            x=agg_val['x'], y=agg_val['mean'], mode='lines',
            name='Validation', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=agg_val['x'], y=agg_val['y_upper'], mode='lines',
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=agg_val['x'], y=agg_val['y_lower'], mode='lines',
            fill='tonexty', fillcolor='rgba(255,165,0,0.1)',
            line=dict(width=0), showlegend=False
        ))

    fig.update_layout(
        title=plot_title,
        xaxis_title='Generation',
        yaxis_title=('Log ' if var == 'size' and logarithmic else '') + ('RMSE' if var == 'rmse' else 'Program Size'),
        height=500, width=800,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
    )
    fig.update_yaxes(range=[0, None])
    fig.show()


def make_slim_evolution_plots(n_rows, n_cols, slim_versions, df_log, plot_title, var='rmse'):
    """
    Create and display a grid of evolution plots for multiple SLIM variations.
    Each subplot shows the mean and ±1 standard deviation envelope of a given metric (`var`)
    over generations (x-axis).

    Parameters
    ----------
    n_rows : int
        Number of rows in the subplot grid.
    n_cols : int
        Number of columns in the subplot grid.
    slim_versions : list
        List of SLIM versions to be plotted. Each identifier is used to filter the corresponding data from `df_log`.
    df_log : pandas.DataFrame
        DataFrame containing the log data for all runs. It must follow a specific structure:
        - Column 0: SLiM version
        - Column 4: Generation
        - Column 5: Train RMSE
        - Column 8: Validation RMSE
        - Column 9: Program size
    plot_title : str
        Title to display above the full grid of plots.
    var : str, optional
        The variable to plot.
        Either `'rmse'` (default) to show training and validation RMSE,
        or `'size'` to show the evolution of program size.

    Returns
    -------
    None
        Displays an interactive Plotly figure with the plots laid out in a grid.

    Notes
    -----
    - The envelope (±1 std) is visualized as a shaded area.
    - When `var='rmse'`, both training and validation curves are shown.
    """
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        subplot_titles=[f'{i}' for i in slim_versions],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    
    for i, sv in enumerate(slim_versions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = i == 0
        
        # Plot data
        df_plot = pd.DataFrame({
            'x': df_log[df_log[0]==sv].iloc[:, 4],
            'rmse': df_log[df_log[0]==sv].iloc[:, 5],
            'rmse_val': df_log[df_log[0]==sv].iloc[:, 8],
            'size': df_log[df_log[0]==sv].iloc[:, 9]
        })
        agg = df_plot.groupby('x')[var].agg(['mean', 'std']).reset_index()
        agg['y_upper'] = agg['mean'] + agg['std']
        agg['y_lower'] = agg['mean'] - agg['std']
        agg.loc[agg['y_lower'] < 0, 'y_lower'] = 0
    
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['mean'],
            mode='lines',
            name='Train' if var=='rmse' else 'Size',
            line=dict(color='blue'),
            showlegend=show_legend
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['y_upper'],
            mode='lines',
            name='+1 std Train',
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=agg['x'],
            y=agg['y_lower'],
            mode='lines',
            name='-1 std Train',
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col)

        if var=='rmse':
            agg = df_plot.groupby('x')['rmse_val'].agg(['mean', 'std']).reset_index()
            agg['y_upper'] = agg['mean'] + agg['std']
            agg['y_lower'] = agg['mean'] - agg['std']
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['mean'],
                mode='lines',
                name='Validation',
                line=dict(color='orange'),
                showlegend=show_legend
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['y_upper'],
                mode='lines',
                name='+1 std Val',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=agg['x'],
                y=agg['y_lower'],
                mode='lines',
                name='-1 std Val',
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.1)',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)
        
    fig.update_layout(
        title_text=plot_title,
        xaxis_title='',
        yaxis_title='',
        height=700,
        width=1100,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    fig.update_yaxes(range=[0, None])
    fig.show()


def plot_features(data, features, num_columns=3, plot_type='Histogram', plot_title='Feature Distributions'):
    """
    Plot feature distributions as either histograms or box plots using Plotly.

    Args:
        data (pd.DataFrame): Input data containing the features.
        features (list): List of column names to plot.
        num_columns (int): Number of columns in the subplot grid.
        plot_type (str): Either 'Histogram' or 'Box' to select plot type.
        plot_title (str): Title for the full figure.
    """
    plot_mapping = {
        'Histogram': go.Histogram,
        'Box': go.Box
    }

    if plot_type not in plot_mapping:
        raise ValueError("plot_type must be 'Histogram' or 'Box'")

    PlotClass = plot_mapping[plot_type]

    num_features = len(features)
    num_rows = (num_features + num_columns - 1) // num_columns
    fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // num_columns + 1
        col = i % num_columns + 1
        values = data[feature].dropna()

        if plot_type == 'Histogram':
            trace = PlotClass(
                x=values,
                name=feature,
                marker_color='black',
                opacity=1,
                showlegend=False
            )
        else:  # Box plot
            trace = PlotClass(
                y=values,
                name=feature,
                marker=dict(color='black'),
                boxmean='sd',
                showlegend=False
            )

        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        height=300 * num_rows,
        width=1100,
        title_text=plot_title,
        bargap=0.2
    )
    fig.show()

def print_slim_individual(nested_list, slim_version, level=0):
    
    # Set up guards for slim version
    has_add  = '+' in slim_version
    has_sig1 = 'SIG1' in slim_version
    has_sig2 = 'SIG2' in slim_version
    has_abs  = 'ABS'  in slim_version

    indent = ' ' * (level * 2)

    # Iter the nested_list
    for item in nested_list:
        if isinstance(item, tuple):
            print(indent + str(item))

        elif isinstance(item, list):
            
            # Build the expression
            str_1 = indent + '+ ' if has_add else indent + '* (1 + '
            str_3 = '' if has_add else ')'

            if has_sig1:
                str_2 = f"{item[2]} * [ 2 * Sig({item[1].structure}) - 1 ]"
            elif has_sig2:
                str_2 = f"{item[3]} * [ Sig({item[1].structure}) - Sig({item[2].structure}) ]"
            elif has_abs:
                str_2 = f"{item[2]} * [ 1 - 2 / ( 1 + |{item[1].structure}| ) ]"
            else:
                str_2 = '<?>'

            print(str_1 + str_2 + str_3)

        else:
            print(indent + str(item))
