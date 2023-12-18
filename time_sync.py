import ntplib
import time

class TimeSync:
    def __init__(self):
        self.ntp_client = ntplib.NTPClient()
        self.start_time = 0

    def sync_time(self):
        try:
            response = self.ntp_client.request('pool.ntp.org')
            current_time = response.tx_time
            self.start_time = time.time() - current_time
            formatted_time = time.strftime("%H:%M", time.localtime(current_time))
            return formatted_time
        except Exception as e:
            print(f"Ошибка Синхронизации Времени: {e}")
            self.start_time = time.time()
            return "Ошибка синхронизации времени"

    def get_current_time(self):
        self.sync_time()
        return time.time() - self.start_time