import csv

dataType = ["Timestamp","Plant_ID","Soil_Moisture","Ambient_Temperature","Soil_Temperature","Humidity","Light_Intensity","Soil_pH","Nitrogen_Level","Phosphorus_Level","Potassium_Level","Chlorophyll_Content","Electrochemical_Signal","Plant_Health_Status"]
#X = {"Timestamp": "2024-10-03 10:54:53.407995", "Plant_ID": 1, "Soil_Moisture": 27.521108772254976, "Ambient_Temperature": 22.24024536256306, "Soil_Temperature": 21.900435355069522, "Humidity": 55.291903895088865, "Light_Intensity": 556.172805131218, "Soil_pH": 5.581954516265902, "Nitrogen_Level": 10.003649716693408, "Phosphorus_Level": 45.80685202827101, "Potassium_Level": 39.0761990273964, "Chlorophyll_Content": 35.703005710811865, "Electrochemical_Signal": 0.9414021464707312, "Plant_Health_Status": "High Stress"}

PlantData = []
data10 = []
data16 = []
data22 = []
data04 = []

with open("plant_health_data.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        copyOfDataType = {}
        for i in range(len(dataType)):
            if i == 0:
                time_str = row[0].split()[1]
                hour = int(time_str.split(":")[0])
                copyOfDataType["Timestamp"] = hour
            elif i < len(row):
                copyOfDataType[dataType[i]] = row[i]

        if copyOfDataType["Timestamp"] == 10:
            data10.append(copyOfDataType)
        elif copyOfDataType["Timestamp"] == 16:
            data16.append(copyOfDataType)
        elif copyOfDataType["Timestamp"] == 22:
            data22.append(copyOfDataType)
        elif copyOfDataType["Timestamp"] == 4:
            data04.append(copyOfDataType)
        del copyOfDataType["Timestamp"]
        del copyOfDataType["Plant_ID"]
        PlantData.append(copyOfDataType)
