import sys
from pathlib import Path
import csv
from functools import lru_cache

sys.path.append(str(Path(__file__).resolve().parent.parent))

from os import listdir
from os.path import isfile, join
from nicegui import ui

from visual.emotion_timeline.emotion_timeline_ui import render_emotion_timeline
from visual.emotion_pies.visual_pie_ui import render_emotion_pie

STATS_PATH = Path('../resources/transcripts_stats')
CAST_PATH = Path('../resources/cast_folder')

CAST_MEMBERS = [f for f in listdir(CAST_PATH) if isfile(join(CAST_PATH, f))]
MEMBERS = ['all'] + sorted([f.replace('.png', '') for f in CAST_MEMBERS])


def root():
    ui.sub_pages({
        '/': table_page,
        '/new': new_page,
    }).classes('w-full')


def table_page():
    ui.table(rows=[{'name': 'Test'}]).on(
        'row-click',
        lambda e: ui.navigate.to('/new'),
    )


@lru_cache(maxsize=64)
def collect_data_cached(member: str):
    collected_rows = []

    for csv_file in STATS_PATH.glob('*.csv'):
        episode = int(csv_file.stem.split('_')[0])

        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                row_member = row.get('speaker', '').strip().lower()

                if member == 'all':
                    if row_member != 'all':
                        row['_source_file'] = csv_file.name
                        row['episode'] = episode
                        collected_rows.append(row)
                elif row_member == member.lower():
                    row['_source_file'] = csv_file.name
                    row['episode'] = episode
                    collected_rows.append(row)

    collected_rows.sort(key=lambda row: (int(row['episode']), row['speaker']))
    return tuple(tuple(item.items()) for item in collected_rows)


def collect_data(member: str) -> list[dict]:
    return [dict(row) for row in collect_data_cached(member)]


def new_page():
    with ui.row().classes('w-full no-wrap'):

        with ui.column().classes('w-1/4 bg-gray-100 p-4 h-screen'):
            ui.label('Cast Members').classes('text-lg font-bold')
            sidebar = ui.column().classes('w-full')

        with ui.column().classes('w-3/4 p-4') as main_content:
            ui.label('Main Content').classes('text-xl')
            ui.label('Select a cast member from the left.')

        def show_member(member: str):
            data = collect_data(member)

            main_content.clear()

            with main_content:
                title = 'Showing all members' if member == 'all' else f'Selected: {member}'
                ui.label(title).classes('text-2xl font-bold')

                if not data:
                    ui.label('No data found.')
                    return

                stats_episodes = sorted({int(row['episode']) for row in data})

                video_results_dir = Path('../results/video')
                visual_episodes = sorted(
                    int(path.stem.replace('episode', '').replace('_results', ''))
                    for path in video_results_dir.glob('episode*_results.csv')
                )

                available_episodes = [ep for ep in stats_episodes if ep in visual_episodes]

                with ui.tabs().classes('w-full') as tabs:
                    stats_tab = ui.tab('Stats')
                    timeline_tab = ui.tab('Emotion Timeline')
                    pie_tab = ui.tab('Engagement / Pie')

                with ui.tab_panels(tabs, value=stats_tab).classes('w-full'):
                    with ui.tab_panel(stats_tab):
                        ui.label(f'Rows found: {len(data)}')

                        columns = [
                            {'name': 'episode', 'label': 'Episode', 'field': 'episode'},
                            {'name': 'speaker', 'label': 'Speaker', 'field': 'speaker'},
                            {'name': 'turns', 'label': 'Turns', 'field': 'turns'},
                            {'name': 'total_sec', 'label': 'Total Seconds', 'field': 'total_sec'},
                            {'name': 'avg_turn_duration', 'label': 'Avg Turn Duration', 'field': 'avg_turn_duration'},
                            {'name': '_source_file', 'label': 'Source File', 'field': '_source_file'},
                        ]

                        ui.table(columns=columns, rows=data).classes('w-full')

                    with ui.tab_panel(timeline_tab):
                        if not available_episodes:
                            ui.label('No timeline episodes available.')
                        else:
                            with ui.row().classes('items-end gap-3 mb-4'):
                                timeline_select = ui.select(
                                    options=available_episodes,
                                    value=available_episodes[-1],
                                    label='Episode',
                                ).classes('w-48')

                                timeline_button = ui.button('Generate')

                            timeline_container = ui.column().classes('w-full')

                            def generate_timeline():
                                timeline_container.clear()
                                with timeline_container:
                                    render_emotion_timeline(member, int(timeline_select.value))

                            timeline_button.on_click(generate_timeline)

                    with ui.tab_panel(pie_tab):
                        if not available_episodes:
                            ui.label('No pie episodes available.')
                        else:
                            with ui.row().classes('items-end gap-3 mb-4'):
                                pie_select = ui.select(
                                    options=available_episodes,
                                    value=available_episodes[-1],
                                    label='Episode',
                                ).classes('w-48')

                                pie_button = ui.button('Generate')

                            pie_container = ui.column().classes('w-full')

                            def generate_pie():
                                pie_container.clear()
                                with pie_container:
                                    render_emotion_pie(member, int(pie_select.value))

                            pie_button.on_click(generate_pie)

        with sidebar:
            for member in MEMBERS:
                ui.button(
                    member,
                    on_click=lambda m=member: show_member(m)
                ).classes('w-full')


ui.run(root)