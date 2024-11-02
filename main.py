from backend.data.DataHandler import DataHandler
from backend.models.rBergomi import rBergomi
from backend.models.NNCalibrator import NeuralNetworkPricer
from frontend.DashBoard import Dashboard
from frontend.SuperDash import VolatilityDashboard
from Statistics import Statistics
import logging

class VolaSurfer:

    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.dh = DataHandler()
        self.db = VolatilityDashboard()

    def main(self):
        try:
            df = self.dh.parse_file(self.file_path)
            df = self.dh.get_binned_data(df)
            logging.info("Data loaded successfully")

            stats = Statistics(df)
            summary = stats.generate_summary()
            print(summary)

            self.db.display(df)

            model = rBergomi(**self.config['rBergomi'])
            calibrated_model = NeuralNetworkPricer(model, df)

            # Further processing...

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    file_path = "Data/SPY_Options_log.txt"
    config = {
        'rBergomi': {
            'n': 100,  # steps per year
            'N': 30000,  # paths
            'T': 1.0,  # maturity
            'a': -0.4  # alpha
        }
    }
    surfer = VolaSurfer(file_path, config)
    surfer.main()