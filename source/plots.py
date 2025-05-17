import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_evolution_plots(n_rows, n_cols, slim_versions, df_log, plot_title, var='rmse'):
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
