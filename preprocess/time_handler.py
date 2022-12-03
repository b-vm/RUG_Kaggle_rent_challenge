import re
from datetime import datetime, timedelta
from preprocess.util import is_nan

from logger import log


def transform_postedagostamp_to_timestamp(posted_ago_stamp, scrape_time):
    if is_nan(posted_ago_stamp):
        return posted_ago_stamp

    try:
        current_time = datetime.strptime(posted_ago_stamp, "%d %b '%y")
        return (scrape_time - (scrape_time - current_time)).timestamp()
    except ValueError:
        pass

    try:
        current_time = datetime.strptime(posted_ago_stamp, "%d %b")
        current_time = current_time.replace(year=2020)
        return (scrape_time - (scrape_time - current_time)).timestamp()
    except ValueError:
        pass

    week = re.search("(\\d)w", posted_ago_stamp)
    if week != None:
        return (scrape_time - (scrape_time - timedelta(weeks=int(week.group(1))))).total_seconds()

    day = re.search("(\\d)d", posted_ago_stamp)
    if day != None:
        return (scrape_time - (scrape_time - timedelta(days=int(day.group(1))))).total_seconds()

    hour = re.search("(\\d)h", posted_ago_stamp)
    if hour != None:
        return (scrape_time - (scrape_time - timedelta(hours=int(hour.group(1))))).total_seconds()

    return (scrape_time - timedelta(days=365)).timestamp()


def find_scrape_time(df):
    best_scrape_time = datetime.strptime(df['lastSeenAt'].iloc[0].split('.')[0], "%Y-%m-%d %H:%M:%S")
    for value in df['lastSeenAt']:
        value = value.split('.')[0]
        value = value.split('+')[0]
        scrape_time = datetime.strptime(value.split('.')[0], "%Y-%m-%d %H:%M:%S")
        if scrape_time > best_scrape_time:
            best_scrape_time = scrape_time
    return best_scrape_time

def preprocess_posted_ago(df):
    column = 'postedAgo'
    log.info(f"Running preprocessing on time column '{column}'...")

    # Estimation of the scrape time
    scrape_time = find_scrape_time(df)

    df[column] = df[column].apply(lambda x: transform_postedagostamp_to_timestamp(x, scrape_time))

    log.info(f"Finished preprocessing on time column '{column}'...")
    return df

def preprocess_raw_availability(df):
    column = 'rawAvailability'

    df['minRawAvailability'] = datetime(2019,1,1).timestamp()
    df['maxRawAvailability'] = datetime(2020,12,1).timestamp()

    for idx, value in enumerate(df[column]):
        availabilities = value.split(' - ')

        if availabilities[0] != "Indefinite period":
            df.iloc[idx, df.columns.get_loc("minRawAvailability")] = datetime.strptime(availabilities[0], "%d-%m-'%y").timestamp()
        if availabilities[1] != "Indefinite period":
            df.iloc[idx, df.columns.get_loc("maxRawAvailability")] = datetime.strptime(availabilities[1], "%d-%m-'%y").timestamp()

    df = df.drop(column, axis="columns")

    return df

def columns_to_timestamp(df, columns):
    for column in columns:
        log.info(f"Running preprocessing on time column '{column}'...")
        df[column] = df[column].apply(lambda x: datetime.strptime(x.split('.')[0].split('+')[0], "%Y-%m-%d %H:%M:%S").timestamp())
        log.info(f"Finished preprocessing on time column '{column}'...")
    return df
