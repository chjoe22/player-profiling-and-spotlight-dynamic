import sys
from pathlib import Path
import csv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from os import listdir
from os.path import isfile, join
from nicegui import ui

STATS_PATH = Path('../resources/transcripts_stats')
CAST_PATH = Path('../resources/cast_folder')

CAST_MEMBERS = [f for f in listdir(CAST_PATH) if isfile(join(CAST_PATH, f))]
MEMBERS = ['all'] + [f.replace('.png', '') for f in CAST_MEMBERS]

def root():
    ui.sub_pages({
        '/': table_page,
        '/map/{lat}/{lon}': map_page,
        '/new': new_page
    }).classes('w-full')

def table_page():
    ui.table(rows=[
        {'name': 'New York', 'lat': 40.7119, 'lon': -74.0027},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
        {'name': 'Tokyo', 'lat': 35.6863, 'lon': 139.7722},
    ]).on('row-click', lambda e: ui.navigate.to(f'/map/{e.args[1]["lat"]}/{e.args[1]["lon"]}'))

    ui.table(rows=[
        {'name': 'Test'},
    ]).on('row-click', lambda e: ui.navigate.to('/new'))

def map_page(lat: float, lon: float):
    ui.leaflet(center=(lat, lon), zoom=10)
    ui.link('Back to table', '/')

def new_page():
    with ui.row().classes('w-full no-wrap'):

        def collect_data(member: str) -> list[dict]:
            collected_rows = []

            for csv_file in STATS_PATH.glob('*.csv'):
                with open(csv_file, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        row_member = row.get('speaker', '').strip().lower()

                        if member == 'all':
                            if row_member != 'all':
                                row['_source_file'] = csv_file.name
                                collected_rows.append(row)
                        elif row_member == member.lower():
                            row['_source_file'] = csv_file.name
                            collected_rows.append(row)

            return collected_rows

        def show_member(member: str):
            data = collect_data(member)

            main_content.clear()
            with main_content:
                if member == 'all':
                    ui.label('Show all members').classes('text-xl')
                else:
                    ui.label(f'Selected: {member}').classes('text-xl')

                ui.label(f'Rows found: {len(data)}')

                if not data:
                    ui.label('No data found.')
                    return

                columns = [
                    {'name': key, 'label': key, 'field': key}
                    for key in data[0].keys()
                ]

                ui.table(columns=columns, rows=data).classes('w-full')

        with ui.column().classes('w-1/4 bg-gray-100 p-4 h-screen'):
            ui.label('Cast Members').classes('text-lg font-bold')

            for member in MEMBERS:
                ui.button(
                    member,
                    on_click=lambda m=member: show_member(m)
                ).classes('w-full')

        with ui.column().classes('w-3/4 p-4') as main_content:
            ui.label('Main Content').classes('text-xl')
            ui.label('Select a cast member from the left.')

ui.run(root)