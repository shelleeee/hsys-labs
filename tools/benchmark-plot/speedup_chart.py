import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import dry

PATH_TO_CHART = f"speedup_chart_{dry.timestamp()}.html"


def calc_speedup(
    complexity: dict, benchmark_target: str, benchmark_reference: str, cpu_time=False
) -> pd.DataFrame:
    target_df = complexity[benchmark_target]
    reference_df = complexity[benchmark_reference]

    target_df["reference"] = reference_df["benchmark"].values

    time_key = "cpu_time" if cpu_time else "real_time"
    target_df["speedup"] = reference_df[time_key].values / target_df[time_key].values

    return target_df


def make_speedup_chart(
    target_df: pd.DataFrame,
    path_to_chart=PATH_TO_CHART,
    cpu_time=False,
    width=1000,
    height=600,
    xaxis_log=True,
    yaxis_log=True,
    dark=False,
    baseline=False,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target_df["size"],
            y=target_df["speedup"],
            mode="lines+markers",
            marker=dict(size=6),
            name="Speedup"
        )
    )

    if baseline:
        fig.add_trace(
            go.Scatter(
                x=target_df["size"],
                y=[1] * len(target_df),
                mode="lines",
                line=dict(dash='dash', color='red'),
                name="Baseline"
            )
        )

    title = f"Speedup: {target_df['benchmark'].values[0]} vs {target_df['reference'].values[0]}"

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="N",
        yaxis_title="Speedup",
        xaxis_type="log" if xaxis_log else "linear",
        yaxis_type="log" if yaxis_log else "linear",
        hovermode="x unified",
        template="plotly_dark" if dark else "plotly_white",
        width=width,
        height=height,
        autosize=False,
    )

    fig.write_html(path_to_chart)
    print(f"The chart file has been saved to {path_to_chart}.")

    return fig


def main(argv):
    argparser = dry.make_default_argparser()

    argparser.add_argument(
        "-c",
        "--chart",
        type=str,
        default=str(PATH_TO_CHART),
        help="Output path for the chart file",
    )

    argparser.add_argument(
        "-r",
        "--reference",
        required=True,
        type=str,
        help="Reference banchmark",
    )

    argparser.add_argument(
        "-t",
        "--target",
        required=True,
        type=str,
        help="Target banchmark",
    )

    argparser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Add baseline (y = 1)",
    )

    args = argparser.parse_args(argv)

    target_df = calc_speedup(
        dry.parse_complexity_many_files([Path(p) for p in args.json]),
        args.target,
        args.reference,
    )

    dry.show_chart(
        make_speedup_chart(
            target_df,
            args.chart,
            args.cpu,
            args.width,
            args.height,
            args.xlog,
            args.ylog,
            args.dark,
            args.baseline,
        ),
        args.chart,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
