from datetime import datetime

start_date_str = "2022-08-25 00:00:00.000000"
delivery_date_str = "2022-09-03 00:00:00.000000"

start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S.%f")
delivery_date = datetime.strptime(delivery_date_str, "%Y-%m-%d %H:%M:%S.%f")

time_span = delivery_date - start_date
time_span_seconds = time_span.total_seconds()
print(f"Time span: {time_span_seconds} seconds")
