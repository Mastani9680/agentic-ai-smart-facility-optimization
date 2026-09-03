import pandas as pd
import os


class EnergyAgent:

    def __init__(self):
        self.energy_file = "data/energy/energy_data.csv"
        self.anomaly_file = "results/anomaly/detected_anomalies.csv"
        self.forecast_file = "results/forecast/energy_forecast.csv"

    # ==========================================
    # LOAD ENERGY DATA
    # ==========================================

    def load_energy_data(self):

        df = pd.read_csv(self.energy_file)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    # ==========================================
    # ENERGY ANALYSIS
    # ==========================================

    def analyze_energy(self):

        df = self.load_energy_data()

        total_energy = df["total_energy"].sum()
        average_energy = df["total_energy"].mean()
        maximum_energy = df["total_energy"].max()

        return {
            "total_energy": round(total_energy, 2),
            "average_energy": round(average_energy, 2),
            "maximum_energy": round(maximum_energy, 2)
        }

    # ==========================================
    # ANOMALY ANALYSIS
    # ==========================================

    def analyze_anomalies(self):

        if not os.path.exists(self.anomaly_file):

            return {
                "anomalies": 0,
                "anomaly_percentage": 0
            }

        anomalies = pd.read_csv(self.anomaly_file)

        total_records = len(self.load_energy_data())

        anomaly_count = len(anomalies)

        percentage = (anomaly_count / total_records) * 100

        return {
            "anomalies": anomaly_count,
            "anomaly_percentage": round(percentage, 2)
        }

    # ==========================================
    # FORECAST ANALYSIS
    # ==========================================

    def analyze_forecast(self):

        if not os.path.exists(self.forecast_file):

            return {
                "forecast_available": False,
                "average_forecast": 0
            }

        forecast = pd.read_csv(self.forecast_file)

        average_forecast = forecast["predicted_energy"].mean()

        return {
            "forecast_available": True,
            "average_forecast": round(average_forecast, 2)
        }
    # ==========================================
    # GENERATE ENERGY RECOMMENDATIONS
    # ==========================================

    def generate_recommendations(self):

        df = self.load_energy_data()

        recommendations = []

        # Average values
        avg_occupancy = df["occupancy"].mean()
        avg_hvac = df["hvac_energy"].mean()
        avg_lighting = df["lighting_energy"].mean()
        avg_equipment = df["equipment_energy"].mean()

        # --------------------------------------
        # HVAC recommendation
        # --------------------------------------

        if avg_hvac > 500 and avg_occupancy < 100:

            recommendations.append(
                "HVAC consumption is high compared with occupancy. "
                "Review HVAC schedules during low-occupancy periods."
            )

        else:

            recommendations.append(
                "HVAC consumption is within the expected range."
            )

        # --------------------------------------
        # Lighting recommendation
        # --------------------------------------

        if avg_lighting > 100:

            recommendations.append(
                "Lighting consumption is relatively high. "
                "Consider optimizing lighting schedules in low-occupancy areas."
            )

        else:

            recommendations.append(
                "Lighting consumption is within the expected range."
            )

        # --------------------------------------
        # Equipment recommendation
        # --------------------------------------

        if avg_equipment > 100:

            recommendations.append(
                "Equipment energy consumption is high. "
                "Check unnecessary equipment operation."
            )

        else:

            recommendations.append(
                "Equipment energy consumption is within the expected range."
            )

        # --------------------------------------
        # General recommendation
        # --------------------------------------

        recommendations.append(
            "Continue monitoring energy anomalies and investigate "
            "repeated high-consumption events."
        )

        return recommendations

    # ==========================================
    # GENERATE ENERGY STATUS
    # ==========================================

    def generate_status(self):

        energy = self.analyze_energy()
        anomaly = self.analyze_anomalies()
        forecast = self.analyze_forecast()

        if anomaly["anomaly_percentage"] >= 5:

            status = "HIGH ENERGY RISK"

        elif anomaly["anomaly_percentage"] >= 3:

            status = "MODERATE ENERGY RISK"

        else:

            status = "NORMAL"

        return {
            "energy": energy,
            "anomaly": anomaly,
            "forecast": forecast,
            "status": status
        }

    # ==========================================
    # DISPLAY AGENT REPORT
    # ==========================================

    def run(self):

        result = self.generate_status()
        recommendations=self.generate_recommendations()

        print("=" * 60)
        print("SMART FACILITY ENERGY AGENT")
        print("=" * 60)

        print("\nENERGY ANALYSIS")
        print("-" * 40)

        print(
            "Total Energy:",
            result["energy"]["total_energy"]
        )

        print(
            "Average Energy:",
            result["energy"]["average_energy"]
        )

        print(
            "Maximum Energy:",
            result["energy"]["maximum_energy"]
        )

        print("\nANOMALY ANALYSIS")
        print("-" * 40)

        print(
            "Detected Anomalies:",
            result["anomaly"]["anomalies"]
        )

        print(
            "Anomaly Percentage:",
            result["anomaly"]["anomaly_percentage"],
            "%"
        )

        print("\nFORECAST")
        print("-" * 40)

        if result["forecast"]["forecast_available"]:

            print(
                "Average Forecasted Energy:",
                result["forecast"]["average_forecast"]
            )

        else:

            print("Forecast not available")

        print("\nENERGY STATUS")
        print("-" * 40)

        print(result["status"])
        print("\nENERGY RECOMMENDATIONS")
        print("-" *40)
        for i, recommendation in enumerate(
            recommendations,
            start=1
            ):
            print(f"{i}. {recommendation}")

        print("\n" + "=" * 60)
        print("ENERGY AGENT COMPLETED")
        print("=" * 60)


# ==========================================
# RUN ENERGY AGENT
# ==========================================

if __name__ == "__main__":

    agent = EnergyAgent()

    agent.run()