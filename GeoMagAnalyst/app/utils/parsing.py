import pandas as pd
import io


async def parse_csv(file):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    # ожидаемый минимум
    # timestamp | value

    return df