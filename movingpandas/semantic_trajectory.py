from shapely.geometry import Point
from datetime import datetime, timedelta
from meteostat import Hourly, Point
from .trajectory import Trajectory
from.trajectory_stop_detector import TrajectoryStopDetector
import pandas as pd
import calendar
import requests
from geopy.geocoders import Nominatim
from functools import partial
import folium
import base64
from functools import reduce
import numpy as np
from owlready2 import *
from pandas._libs.tslibs.timestamps import Timestamp 
import haversine as hs


class SemanticTrajectory(Trajectory):
  """
    Reconstruct Trajectory from image metadata taken by flying object (e.g. drone)

    Parameters
    ----------
    trajectory : Trajectory
      Is a raw trajectory (contains only spatial and temporal data) reconstructed from images 
      or created by using a csv file
    """
  def __init__(self, traj, traj_id, drone, mission, flight, source):
    super().__init__(traj, traj_id)
    self.trajectory_df = self.df.reset_index()
    self.source = source
    path = os.path.join("scaled", self.source)
    path = os.path.join("photosets", path)
    self.source_path = os.path.abspath(path)
    self.drone = drone
    self.mission = mission
    self.flight = flight
    self.weather_data = pd.DataFrame()
    self.points_of_interest = pd.DataFrame()
    self.segments = [(0, len(self.trajectory_df)-1)]
    self.stops = None


  def _get_ground_data(self, lat, lon, date_time, lang, api_key):
    """
      Get the ground weather data based on lat, lon, timestamp 
      using the OpenWeather api.

        Parameters
        ----------
        lat :
          the latitude of the point
        lon :
          the longitude of the point
        date_time :
          the timestamp of the point
        lang:
          the language of the returned data
        api_key :
          the user's api key for the OpenWeather api

        
        Return
        ---------
        The fetched data
    """
    
    timestamp = datetime.datetime.strptime(date_time, '%y:%m:%d %H:%M:%S')
    dt = calendar.timegm(timestamp.utctimetuple())
    
    api_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={dt}&lang={lang}&appid={api_key}&units=metric"
    response = requests.get(api_url)
    weather_data = response.json()
    return pd.Series([weather_data['data'][0]['dew_point'], \
      weather_data['data'][0]['temp'], \
      weather_data['data'][0]['temp'], \
      weather_data['data'][0]['pressure'], \
      weather_data['data'][0]['wind_deg'], \
      weather_data['data'][0]['wind_deg'], \
      weather_data['data'][0]['wind_speed'], \
      weather_data['data'][0]['wind_speed']])
  

  def _get_air_data(self, point, start, end):
    """
      Get the air weather data per hour based on lat, lon, timestamp and altitude
      using the meteostat web api

        Parameters
        ----------
        point :
          instance of class Point(latitude, longitude, altitude)
        start :
          the start timestamp of the point
        end :
          the end timestamp plus one hour
        
        Return
        ---------
        The fetched data
    """
    data = Hourly(point, start, end)
    data = data.fetch()

    return pd.Series([
        float(data['dwpt']), \
        float(data['temp']), \
        float(data['temp']), \
        float(data['pres']), \
        float(data['wdir']), \
        float(data['wdir']), \
        float(data['wspd']), \
        float(data['wspd'])
    ])


  def get_weather_data(self, api_key = None, lang="en", mode="ground"):
    """
      Get the weather data for each Recording Point of the trajectory, 
      using two web services (openweathermap, meteostat). The user can 
      fetch either ground weather data or air weather data. The air weather data
      are only available using the meteostat web service. If no weahter data for
      the given altitude is available then the ground weather data are fetch. 
      To use the openweathermap, the user must create a valid api_key. 

      openweathermap fetch the ground weahter data based on lat, lon, and timestamp.
      meteostat fetch the (ground/air) data based on lat, lon, timestamp and altitude

        Parameters
        ----------
        api_key :
          api_key necessary for the openweathermap web service (Default = None)  
        lang :
          The language of the data to be returned (Default = en -> English)
        mode :
          determines the web service.
          ground: Fetch the ground weather conditions using the openweathermap if a valid api_key exists.
                  Otherwise, the ground weather data are fetched from the meteostat.
          air: Fetch the air weather data using the meteostat web service.
        
        Examples
        --------
        >>> api_key = "........"
        >>> traj.get_weather_data(api_key=api_key, mode="ground")
    """
    data = self.trajectory_df.filter(['t', 'alt', 'geometry'], axis=1)
    self.weather_data = data

    weather_attributes = ['dew_point', 'max_temperature', 'min_temperature', 'pressure', 'wind_direction_max', 'wind_direction_min', 'wind_speed_max', 'wind_speed_min']
   
    if api_key is not None and mode == "ground":
      try:
        for attr in weather_attributes:
          self.weather_data[attr] = None
      
        data = self.weather_data.apply(
          lambda row: self._get_ground_data(row['geometry'].y, row['geometry'].x, row['t'].strftime("%y:%m:%d %H:%M:%S"), lang, api_key),
          axis = 1
        )
        self.weather_data[weather_attributes] = data
        print("Ground weather data were fetched by https://openweathermap.org/")
      except:
        print("Please use a valid api key. Visit https://openweathermap.org/ for more info.") 
        
    else:
      no_air_data = False
      if mode == "air":
        try:
          for attr in weather_attributes:
            self.weather_data[attr] = None
          
          data = self.weather_data.apply(
            lambda row: self._get_air_data(Point(row['geometry'].y, row['geometry'].x, int(row['alt'])), row['t'], row['t'] + timedelta(hours=1)),
            axis = 1
          )
          self.weather_data[weather_attributes] = data
          print("Air weather data were fetched by https://meteostat.net/en/")
        except:
          print("No weather data available for the specific altitude.")
          print("An attempt will be made to recover the ground weather data.")

          no_air_data = True 

      if mode == "ground" or no_air_data:
        try:
          for attr in weather_attributes:
            self.weather_data[attr] = None
          
          data = self.weather_data.apply(
            lambda row: self._get_air_data(Point(row['geometry'].y, row['geometry'].x), row['t'], row['t'] + timedelta(hours=1)),
            axis = 1
          )
          self.weather_data[weather_attributes] = data
          print("Ground weather data were fetched by https://meteostat.net/en/")
        except:
          print("No weather data available for the specific area.")
          print("Please create a valid api key. Visit https://openweathermap.org/ for more info.")

  
  def _get_points(self, lat, lon, user_agents, lg):
    """
      This method fetch the data using the API of OpenStreetMap

      Parameters
      ----------
        lat:
          the latitude of the point  
        lon:
          the longitude of the point
        user_agent:
          the name of the user or the app
        lg:
          the language of the fetched data

      Return
      ----------
      The fetched data as Series
    """
    geolocator = Nominatim(user_agent=user_agents)
    reverse = partial(geolocator.reverse, language=lg)
    location = reverse(f"{lat}, {lon}", exactly_one=True)

    return pd.Series([
      location.raw['place_id'], \
      location.raw['osm_type'], \
      location.raw['display_name'], \
      location.raw['boundingbox'], \
      location.raw['address']
    ])

  
  def get_points_of_interest(self, user_agent="ReconTraj4Drones", lang="en"):
    """
      Get the Point of Interest (POI) of each recording point using
      the OpenStreetMap (OSM) api

      Parameters
      ----------
        user_agent:
          the name of the user or the app  
        lang:
          the language of the fetched data
      
      Examples
      --------
      >>> traj.get_points_of_interest()
    """
    data = self.trajectory_df.filter(['t', 'alt', 'geometry'], axis=1)
    self.points_of_interest = data

    poi_attributes = ['place_id', 'osm_type', 'display_name', 'boundingbox', 'address']
    try:
      for attr in poi_attributes:
        self.points_of_interest[attr] = None

      data = self.points_of_interest.apply(
        lambda row: self._get_points(row['geometry'].y, row['geometry'].x, user_agent, lang),
        axis = 1
        )
      self.points_of_interest[poi_attributes] = data
      print("Point of Interest data were fetched by https://www.openstreetmap.org/")
    except:
      print("No Point of Interest data available for the specific area.")


  def _get_stop_rec_points(self, start, end):
    """
      Finds and returns the recording points included in the stop. The search is based on 
      the given timestamps that determines the beginning and the end of the stop.

      Parameters
      ----------
        start:
          defines the beginning of the stop
        end:
          defines the end of the stop.

      Return
      ------
      The number of the recording points included in the stop.
    """
    return len(self.trajectory_df[(self.trajectory_df['t'] >= start) & (self.trajectory_df['t'] <= end)])


  def stop_detector(self, seconds, max_diameter):
    """
      Detects the returns the stops of the trajectory. The stop are detected using the 
      functionality of MovingPandas. The returned stops are then displayed on the users.
      The user determines the stop duration in seconds and the maximum diameter.

      Parameters
      ----------
        seconds:
          the seconds threshold that determine a stop
        max_diameter:
          the maximum diameter threshold that determines a stop
      
      Return
      ------
      The detected stops of the trajectory

      Examples
      --------
      >>> traj.stop_detector(seconds=15, max_diameter=20)
    """
    detector = TrajectoryStopDetector(self)
    self.stops = detector.get_stop_points(min_duration=timedelta(seconds=seconds), max_diameter=max_diameter)
    if len(self.stops) == 1:
      print(f"{len(self.stops)} stop was found using as paremeters seconds = {seconds} and max diameter = {max_diameter}")
      print("The stop found are the following: ")
    elif len(self.stops) > 1:
      print(f"{len(self.stops)} stops were found using as paremeters seconds = {seconds} and max diameter = {max_diameter}")
      print("The stops found are the following: ")
    else:
      print(f"No stop was found using as paremeters seconds = {seconds} and max diameter = {max_diameter}")
      return 

    stops = self.stops.reset_index()
    for i in range(len(stops)):
      print(f"Stop id: {stops.iloc[i]['stop_id']}")
      print(f"Trajectory id: {stops.iloc[i]['traj_id']}")
      print(f"coordinates: {stops.iloc[i]['geometry'].x, stops.iloc[i]['geometry'].y}")
      print(f"Start time: {stops.iloc[i]['start_time']}", end=", ")
      print(f"End time: {stops.iloc[i]['end_time']}", end=", ")
      print(f"Duration: {stops.iloc[i]['duration_s']} sec")
      print(f"Number of Recording Points: {self._get_stop_rec_points(stops.iloc[i]['start_time'], stops.iloc[i]['end_time'])}")
      print(30 * "-")
    
  
  def show(self, percent=1, tiles = "OpenStreetMap",  show_segments=False, show_stops = False, radius = 10):
    """
      Show the enriched trajectory using OpenStreetMap (OSM). For each enriched trajectory 
      the starting and ending point are depicted using a green and a red point respectively.
      The user can determine the portion of the points to be displayed on the map. Each point is 
      clickable and shows more information about the particular point. If the segmentation process
      has been executed, the user can depict these segments. Similarly, if the stop detection process
      has been executed, the user can depict the stops. Each stop is a clickable object that shows
      more information about the particular stop

      Parameters
      ----------
        percent:
          the percentage of points that will appear on the map (Default = 1, 100%)  
        tiles:
          defines the form of the map. Possible values : Stamen Toner, CartoDB positron, Cartodb dark_matter, 
          Stamen Watercolor, Stamen Terrain, or OpenStreatMap. (Default = "OpenStreatMap") 
        show_segments:
          determines whether the trajectory segments will be displayed or not. (Default = False)
        show_stops:
          determines whether the trajectory stops will be displayed or not. (Default = False)
        radius:
          determines the radius of the circles in the display of stops. (Default = 10)
      
      Examples
      --------
      >>> traj.show(percent=0, tiles="OpenStreetMap",  show_segments=False, show_stops=True, radius=10)
      >>> traj.show(percent=0.2, tiles="OpenStreetMap",  show_segments=True, show_stops=False, radius=10)
    """
    min_points = 2
    total_dfs = [self.trajectory_df, self.points_of_interest, self.weather_data]
    selected_dfs = [df for df in total_dfs if not df.empty]
    semantic_df = reduce(lambda left,right: pd.merge(left,right,on=['t', 'alt', 'geometry'], how='outer'), selected_dfs)
    semantic_df = semantic_df.reset_index()

    colors = ['blue', 'green', 'purple', 'black', 'red', 
              'darkblue', 'darkgreen', 'darkpurple', 
              'darkred', 'gray', 'lightblue', 
              'lightgray', 'lightgreen', 'lightred', 
              'orange', 'pink', 'beige', 'cadetblue'
              ]
    # Show map
    m = folium.Map(location=[semantic_df.iloc[0]['geometry'].y, semantic_df.iloc[0]['geometry'].x], tiles=tiles, width="%100",
        height="%100", zoom_start= 50)
    # Get the coordinates of the dataframe
    coords = semantic_df.apply(
      lambda row: (row['geometry'].y, row['geometry'].x),
        axis = 1
    )

    if not show_segments:
      # Show entire Trajectory
      folium.PolyLine(coords,
                      color= "blue",
                      weight=2,
                      opacity=0.8).add_to(m)


      # Calculation of the points to be displayed
      num_of_points = int(len(semantic_df) * percent)
      if num_of_points <= min_points:
        points = [(0, "green"), (len(semantic_df)-1, "red")]
      else:
        lin_points = np.linspace(0, len(semantic_df)-1, num=num_of_points, endpoint=True)
        points = [(0, "green"), (len(semantic_df)-1, "red")]
        for point in range(1, len(lin_points)-1):
          points.append((int(lin_points[point]), "blue"))
    
    else:
      points = []
      # Show the segmented trajectories 
      for index, segment in enumerate(self.segments):
        # Calculation of the points to be displayed
        num_of_points = int((segment[1] - segment[0] + 1) * percent)
        lin_points = np.linspace(segment[0], segment[1], num=num_of_points, endpoint=True)
        points.append((segment[0], "green"))
        points.append((segment[1], "red"))
        for point in range(1, len(lin_points)-1):
          points.append((int(lin_points[point]), "blue"))
    
        
        coords_list = coords[segment[0]: segment[1]]
        c = colors[index]

        folium.PolyLine(coords_list,
                        color=c,
                        weight=2,
                        opacity=0.8).add_to(m)
    if show_stops:
      stops = self.stops.reset_index()
      for i in range(len(stops)):
        html =f"""
              <p><strong>Stop ID:</strong> {stops.iloc[i]['stop_id']}</p>
              <p><strong>Trajectory ID:</strong> {stops.iloc[i]['traj_id']}</p>
              <p><strong>Coordinates:</strong> ({round(stops.iloc[i]['geometry'].y, 6)}, {round(stops.iloc[i]['geometry'].x, 6)})</p>
              <p><strong>Start time:</strong> {stops.iloc[i]['start_time']}</p>
              <p><strong>End time:</strong> {stops.iloc[i]['end_time']}</p>
              <p><strong>Duration:</strong> {stops.iloc[i]['duration_s']} sec</p>
              <p><strong>Recording Points:</strong> {self._get_stop_rec_points(stops.iloc[i]['start_time'], stops.iloc[i]['end_time'])}</p>
              """
        iframe = folium.IFrame(html=html, width=300, height=250)
        popup = folium.Popup(iframe, max_width=500)
        folium.CircleMarker(location=(stops.iloc[i]["geometry"].y, 
                            stops.iloc[i]["geometry"].x),
                            popup=popup,
                            radius=radius, 
                            tooltip = "Click for more info",
                            fill_color='red').add_to(m)

    for point in points:
      index = point[0]
      color = point[1]
      file = os.path.join(self.source_path, semantic_df.iloc[index]['title'])
      encoded = base64.b64encode(open(file, 'rb').read()).decode()
      html = f"""
              <p><strong>Mission:</strong> {self.mission}</p>
              <p><strong>Flight:</strong> {self.flight}</p>
              <p><strong>Drone:</strong> {self.drone}</p>
              <center><img style="width:200;height:200;" src="data:image/jpeg;base64, {encoded}"></center>"""
      html +=f"""
              <p><strong>Lat:</strong> {round(semantic_df.iloc[index]['geometry'].y, 6)}</p>
              <p><strong>Lon:</strong> {round(semantic_df.iloc[index]['geometry'].x, 6)}</p>
              <p><strong>Timestamp:</strong> {semantic_df.iloc[index]['t']}</p>
              <p><strong>Altitude:</strong> {semantic_df.iloc[index]['alt']}</p>
              <p><strong>Camera Model:</strong> {semantic_df.iloc[index]['camera_model'].replace("_", "")}</p>
              <p><strong>Filename:</strong> {semantic_df.iloc[index]['title']}</p>
              """
      if not self.points_of_interest.empty:
        html +=f"""
              <h3><u>Point of Interest Data</u></h3>
              <p><strong>Place id:</strong> {semantic_df.iloc[index]['place_id']}</p>
              <p><strong>OSM type:</strong> {semantic_df.iloc[index]['osm_type']}</p>
              <p><strong>Display Name:</strong> {semantic_df.iloc[index]['display_name']}</p>
              """
      if not self.weather_data.empty:
        html +=f"""
              <h3><u>Weather Data</u></h3>
              <p><strong>Dew Point:</strong> {semantic_df.iloc[index]['dew_point']}</p>
              <p><strong>Temeprature:</strong> {semantic_df.iloc[index]['min_temperature']}°C - {semantic_df.iloc[index]['max_temperature']}°C</p>
              <p><strong>Pressure:</strong> {semantic_df.iloc[index]['pressure']} hPa</p>
              <p><strong>Wind Direction:</strong> {semantic_df.iloc[index]['wind_direction_min']}° - {semantic_df.iloc[index]['wind_direction_max']}°</p>
              <p><strong>Wind Speed:</strong> {semantic_df.iloc[index]['wind_speed_min']} m/s - {semantic_df.iloc[index]['wind_speed_max']} m/s</p>
               """
   
      iframe = folium.IFrame(html=html, width=300, height=300)
      popup = folium.Popup(iframe, max_width=500)
  
      folium.Marker(
        location=[semantic_df.loc[index]['geometry'].y, semantic_df.loc[index]['geometry'].x],
        popup=popup,
        tooltip = "Click for more info",
        icon=folium.Icon(color=color, icon='info', prefix='fa')
      ).add_to(m)

    return m
  

  def export_to_csv(self):
    """
      Export the enriched trajectory in csv file
    """
    total_dfs = [self.trajectory_df, self.points_of_interest, self.weather_data]
    total_dfs = [df for df in total_dfs if not df.empty]
    semantic_df = reduce(lambda left,right: pd.merge(left,right,on=['t', 'alt', 'geometry'], how='outer'), total_dfs)
    self.semantic_df = semantic_df.reset_index()
    filename = "csv_files/" + "enriched_" + self.source + '.csv'
    semantic_df.to_csv(filename, sep=',', index=False, encoding='utf-8')

  
  def _haversine_distance(self, point1, point2):
    """
      Calculate and return in meters the haversine distance between two points of 
      the enriched trajectory

      Parameters
      ----------
        point1:
          the first point of the trajectory. The point must be a Geometry type 
        point2:
          the second point of the trajectory. The point must be a Geometry type
      
      Returns
      -------
        The distance (in meters) between the given points.
    """
    p1 = [point1.y, point1.x]
    p2 = [point2.y, point2.x]
    return hs.haversine(p1, p2)*1000


  def trajectory_segmentation(self, threshold, attr="altitude"):
    """
      Segment the enriched trajectory based on a criterion and a threshold

      Parameters
      ----------
        threshold:
          the threshold above which segmentation will take place
        attr:
          the attribute on which the segmentation will take place.
          Valid values: 
            altitude => segmentation based on altitude (default)
            time => segmentation based on time
            distance => segmentation based on eucledian distance
      
      Return
      ------
        The segmented trajectory
      
      Examples
      --------
      >>> traj.trajectory_segmentation(threshold=25, attr="distance")
    """
    selection_dict = {
      "altitude" : "alt",
      "time": "t",
      "distance": "geometry"
    }

    selection = selection_dict[attr]
    self.segments.clear()
    start = 0
    for i in range(len(self.trajectory_df)-1):
      if selection == "alt":
        diff = round(abs(self.trajectory_df.iloc[i][selection] - self.trajectory_df.iloc[i+1][selection]),4)
      elif selection == "t":
        diff = abs((self.trajectory_df.iloc[i][selection] - self.trajectory_df.iloc[i+1][selection]).total_seconds() / 60)
      else:
        diff = self._haversine_distance(self.trajectory_df.iloc[i][selection], self.trajectory_df.iloc[i+1][selection])

      if diff > threshold:
        stop = i
        if stop - start == 0:
          continue
        self.segments.append((start, stop))
        start = i+1
      if i == len(self.trajectory_df)-2:    
        stop = len(self.trajectory_df)-1
        self.segments.append((start, stop))
    if len(self.segments) > 1:    
      print(f"{len(self.segments)} segments were created based on {attr} with a threshold of {threshold} meters")
      print("The created segments are the following: ")
    else:
      print(f"{len(self.segments)} segment was created based on {attr} with a threshold of {threshold} meters")
      print("The created segment is the following: ")
    for segment in self.segments:
      print(f"(point-{segment[0]} - point-{segment[1]})")


  def export_to_rdf(self, selected_format="rdfxml"):
    """
      Export the semantic trajectory in rdf triples

      Parameters
      ----------
        selected_format:
          the format of the extracted file
          Valid values: 
            rdfxml => extraction in rdf/xml format (default)
            ntriples => extraction in triplesformat
      
      Returns
      -------
        The ontology with the inserted individuals 

      Example
      -------
      >>> default_world = traj.export_to_rdf(selected_format="ntriples")
    """
    total_dfs = [self.trajectory_df, self.points_of_interest, self.weather_data]
    total_dfs = [df for df in total_dfs if not df.empty]
    semantic_df = reduce(lambda left,right: pd.merge(left,right,on=['t', 'alt', 'geometry'], how='outer'), total_dfs)
    self.semantic_df = semantic_df.reset_index()


    onto_path.append("ontologies")
    datacron = get_ontology('ontologies/TopDatAcronOnto_SSN_FM.owl').load()
    sf = get_ontology("http://schemas.opengis.net/sf/1.0/simple_features_geometries.rdf").load()
    dul = get_ontology("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl").load()
    onto = get_ontology('ontologies/dront.owl').load()


    # drone type instance creation
    uav_code = self.drone.replace(" ", "_").lower()
    uav = onto.Drone(uav_code)
    uav.label = uav_code.replace("_", " ")

    # DronePhotographicCamera type instance creation
    recording_equipment_set = set(self.trajectory_df['camera_model'])
    for eq in recording_equipment_set: 
      equipment_code = eq.replace(" ", "_").lower()
      equipment = onto.DronePhotographicCamera(equipment_code)
      equipment.label = equipment_code.replace("_", " ")
      equipment_parts = eq.split("_")
      equipment.hasMaker = [equipment_parts[0]]
      equipment.hasModel = [equipment_parts[1]]

    # Contect Drone instence with DronePhotographicCamera instance through the carriesEquipment object property
    uav.carriesEquipment = [equipment]

    # Trajectory type instance creation
    trajectories_set = set(self.trajectory_df['trajectory_id'])
    for index, traj in enumerate(trajectories_set): 
        trajectory_code = f"trajectory_{index:03}"
        trajectory = datacron.Trajectory(trajectory_code, namespace = onto)
        trajectory.label = trajectory_code.replace("_", " ")
    
    # Connect Drone instance with Trajectory  instance using the hasTrajectory object property
    uav.hasTrajectory = [trajectory]

    # Mission type instance creation
    mission_code = self.mission.replace(" ", "_").lower()
    mission = onto.Mission(mission_code)
    mission.label = mission_code.replace("_", " ")

    # Flight type instance creation
    flight_code = self.flight.replace(" ", "_").lower()
    flight = sf.Flight(flight_code, namespace = onto)
    flight.label = flight_code.replace("_", " ")

    # Connect Drone instance with Flight instance using the hasFlight object property
    uav.hasFlight = [flight]

    # Connect Mission instance with Flight instance using the includesFlight object property
    mission.includesFlight = [flight]

    # RecordingSegment type instance creation
    segments_dict = {}
    for index, segment in enumerate(self.segments):
      recording_segment_code = f"recording_segment_{index+1}"
      recording_segment = onto.RecordingSegment(recording_segment_code)
      recording_segment.label = recording_segment_code.replace("_", " ")
      segments_dict[segment] = recording_segment

    for row in self.semantic_df.itertuples(index=False):
      # For each point we create Geometry type instace 
      point_code = f"point_{row.index:03}"
      point = datacron.Geometry(point_code, namespace = onto)
      point.label = f"point {row.index:03}"
      point.hasAltitude = [str(row.alt)]
      point.hasLattitude = [str(row.geometry.x)]
      point.hasLongitude = [str(row.geometry.y)]
        
      # Instant type instance creation
      instant_code = f"instant_{row.index:03}"
      instant = datacron.Instant(instant_code, namespace = onto)
      instant.label = instant_code.replace("_", " ")
      instant.hasIntervalDate = [row.t.strftime("%y:%m:%d %H:%M:%S")]
        
      # Connection Instant instance with Geometry instance using the hasTemporalFeature object property
      point.hasTemporalFeature = [instant]

      if not self.weather_data.empty:  
        # WeatherCondition type instance creation
        weather_condition_code = f"weather_condition_{row.index:03}"
        weather_condition = datacron.WeatherCondition(weather_condition_code, namespace = onto)
        weather_condition.label = weather_condition_code.replace("_", " ")
        weather_condition.reportedMaxTemperature = [str(row.max_temperature)]
        weather_condition.reportedMinTemperature = [str(row.min_temperature)]
        weather_condition.reportedDewPoint = [str(row.dew_point)]
        weather_condition.reportedPressure = [str(row.pressure)]
        weather_condition.windDirectionMax = [str(row.wind_direction_max)]
        weather_condition.windDirectionMin = [str(row.wind_direction_min)]
        weather_condition.windSpeedMax = [str(row.wind_speed_max)]
        weather_condition.windSpeedMin = [str(row.wind_speed_max)] 
        
      # RecordingPosition type instance type creation
      recording_position_code = f"recording_position_{row.index:03}"
      recording_position = onto.RecordingPosition(recording_position_code)
      recording_position.label = recording_position_code.replace("_", " ")

      # RecordingSegment instace connection with the RecordingPosition instance using the comprises object property
      for segment in self.segments:
        if segment[0] <= row.index <= segment[1]:
          recording_segment = segments_dict[segment]
          recording_segment.comprises.append(recording_position)

      # Trajectory instance connection with the RecordingSegment instance using the encloses object property
      if len(self.segments) == 1:
        recording_segment = segments_dict[self.segments[0]]
        trajectory.encloses = [recording_segment]
      else:
        for segment in self.segments:
          recording_segment = segments_dict[segment]
          trajectory.encloses.append(recording_segment)
      
      # RecordingPosition instance connection with the WeatherCondition instance using the hasWeatherCondition object property
      if not self.weather_data.empty:
        recording_position.hasWeatherCondition = [weather_condition]
        
      # RecordingPosition instance connection with the Geometry instance using the hasGeometry object property
      recording_position.hasGeometry = [point]
        
      # Photograph type instace creation
      photograph_code = str(row.title)
      photograph_code = photograph_code.replace("_", "")
      photograph_code = photograph_code.replace(".JPG", "")
      photograph = onto.Photograph(photograph_code)
      photograph.label = photograph_code
      photograph.hasSize = [str(row.size)]
      photograph.hasStoragePath = [str(row.storage_path)]
      photograph.hasFormat = [str(row.format)]
        
      # RegionofInterest type instace creation
      if not self.points_of_interest.empty:
        region_of_interest_code = f"region_of_interest_{row.index:03}"
        region_of_interest = onto.Region_of_Interest(region_of_interest_code)
        region_of_interest.label = region_of_interest_code.replace("_", " ")
        region_of_interest.hasAddress = [str(row.display_name)]
        region_of_interest.hasPlaceID = [str(row.place_id)]
        region_of_interest.hasOSMType = [str(row.osm_type)]
        
        # RegionofInterest connection with the Photograph instance using the isRecordedIn object property
        region_of_interest.isRecordedIn = [photograph]
        
      # PhotoShootingEvent type instances creation
      photo_shooting_event_code = f"recording_event_{row.index:03}"
      photo_shooting_event = onto.PhotoShootingEvent(photo_shooting_event_code)
      photo_shooting_event.label = photo_shooting_event_code.replace("_", " ")

      recording_position.hasOccurredEvent = [photo_shooting_event]

      # PhotoShootingEvent connection with Drone using the hasDroneParticipant object property
      photo_shooting_event.hasDroneParticipant = [uav]

      # PhotoShootingEvent connection with RegionofInterest with records object property 
      if not self.points_of_interest.empty:
        photo_shooting_event.records = [region_of_interest]

      # PhotoShootingEvent connection with Photograph using produces object property 
      photo_shooting_event.produces = [photograph]

      # PhotoShootingEvent connection with RecordingPosition using occurs object property 
      photo_shooting_event.occurs = [recording_position]

    onto.save(file =  "ontologies/" + self.source +".xml", format=selected_format)
    return default_world


  def get_num_of_recording_points_sparql(self):
    """
      SPARQL query that returns the number of recording points in the given trajectory 
      This method must be executed after the export_to_rdf() method

      Parameters
      ----------

      Returns
      -------
        The results og the SPARQL query
    
      Example
      -------
      >>> traj.get_num_of_recording_points_sparql()
    """
    return list(default_world.sparql("""
            SELECT (COUNT(?recording_points) AS ?rp) WHERE {
                    ?tr dront:encloses ?rs . 
                    ?rs TopDatAcronOnto_SSN_FM:comprises ?recording_points .
              }
      """))[0][0]


  def get_pois_sparql(self):
    """
      SPARQL query that returns the POIs in the given trajectory 
      This method must be executed after the export_to_rdf() method

      Parameters
      ----------

      Returns
      -------
        The results og the SPARQL query
    
      Example
      -------
      >>> traj.get_pois_sparql()
    """
    return list(default_world.sparql("""
            SELECT ?point_of_interest WHERE {
                  ?tr dront:encloses ?rs . 
                  ?rs TopDatAcronOnto_SSN_FM:comprises ?rp .
                  ?rp dront:hasOccurredEvent ?pse .
                  ?pse dront:records ?point_of_interest .
}
      """))