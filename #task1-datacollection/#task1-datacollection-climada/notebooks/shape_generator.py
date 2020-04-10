from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, Polygon
import argparse
import os.path


def evaluate_input(args):
    ## Evaluate the existance of the file
    if os.path.isfile(args.filename):
        try:
            data = pd.read_csv(args.filename)
            if args.windspeed not in data.columns.to_list():
                print("Unable to find the wind speed column")
                return 0
            if ((data['name'] == args.name).any()) and ((data['year'] == args.year).any()):
                return 1
            else:
                print(f"Unable to find the entries for {args.name} name or {args.year} year")
                return 0
        except IOError as e:
            print("Unable to read the CSV file". e)
            return 0
    else:
        print("Unable to find the file")
        return 0


class Shape_Generator:
    def __init__(self, file_path, cyclone_name, cyclone_year, wind_speed, debug=False):
        """
        Initiate shape generator with the input
        :file_path : path of the filename
        :cyclone_name: name of the cyclone
        :wind_speed: Wind speed column name
        """
        data_df = pd.read_csv(file_path)
        cyclones = data_df[['name',
                                 'year', 'lat', 'long', wind_speed]]
        cyclones_filter = cyclones[(cyclones.year == cyclone_year) & (
                            cyclones.name == cyclone_name)]
        self.cyclone_df = pd.DataFrame(cyclones_filter)
        self.cyclone_name = cyclone_name
        self.cyclone_year = cyclone_year
        self.wind_speed = wind_speed
        self.dest_path = os.path.split(file_path)[0]
        self.debug = debug
        if self.debug:
            print("Dataframe after filtering ", self.cyclone_df.head())
        self.cyclone_gdf = self.__generate_geopandas_df__(self.cyclone_df)
        pass

    @staticmethod
    def __generate_geopandas_df__(df):
        """
        Generate the geopandas dataframe using latitude and longitude
        :df: pandas dataframe with columns lat and long
        :return: Geopandas dataframe
        """
        geometry = [Point(xy) for xy in zip(df.long, df.lat)]
        return GeoDataFrame(df, geometry=geometry)
    
    def generate_center_path(self, generateFile=True):
        """
        Generate the output files for centerpath
        :generateFile : True/False to generate the files using the computed centerpath
        """
        group_gdf = self.cyclone_gdf.groupby(['name', 'year'])[
            'geometry'].apply(lambda x: LineString(x.tolist()))
        if self.debug:
            print(f'Length of linestring dataframe {len(group_gdf)}')
            print(f'First five row of {group_gdf.head()}')
        final_gdf = geopandas.GeoDataFrame(group_gdf, geometry='geometry')
        if self.debug:
            print(final_gdf)
            final_gdf.plot()
            if generateFile:
                plt.show()
            else:
                plt.show(block=False)
        if (final_gdf.crs == None):
            final_gdf.crs = "epsg:4326"
        extension = "centerpath.shp"
        self.final_gdf = final_gdf
        if generateFile:
            self.generate_shapefile(extension)

    def generate_shapefile(self, extension):
        """
        Generate the output files from the geopandas dataframe
        :extension: extension of the output file
        """
        filename = f"{self.cyclone_name}_{self.cyclone_year}_{self.wind_speed}_{extension}"
        if (self.dest_path != ''):
            outfile = os.path.join(self.dest_path, filename)
        else:
            outfile = filename
        if self.debug:
            print(f"outfile is {outfile}")
        self.final_gdf.to_file(filename=outfile, driver="ESRI Shapefile")
    

    def generate_buffered_file(self, radius):
        """
        Generate the buffered output
        :radius The required radius for the buffered output
        """
        # generate the centerpath first and make it buffered
        self.generate_center_path(generateFile=False)
        # If no radius is mentioned
        if radius == 0:
            # compute mean
            buffer_radius = round(self.cyclone_df[self.wind_speed].mean(),2)
            # compute median
            buffer_radius = round(self.cyclone_df[self.wind_speed].median(),2)
            if self.debug:
                print(f'buffered radius = {buffer_radius}')
        # Project to crs that uses meters as distance measure
        cyclone_3395 = self.final_gdf.to_crs('epsg:3395')
        cyclone_3395['geometry'] = cyclone_3395.buffer(buffer_radius*1000)
        buffered_cyclone_gdf = cyclone_3395.to_crs('epsg:4326')
        if self.debug:
            print(buffered_cyclone_gdf.head())
            buffered_cyclone_gdf.plot()
            plt.show()
        self.final_gdf = buffered_cyclone_gdf
        extension = f"{buffer_radius}_buffered.shp"        
        self.generate_shapefile(extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a shape file')
    parser.add_argument("-f", "--filename", dest = "filename", help="Input datafile",required=True)
    parser.add_argument("-n", "--name", dest = "name", help="Cyclone name", required=True)
    parser.add_argument("-y", "--year",dest ="year", help="Cyclone year", type=int, required=True)
    parser.add_argument("-w", "--windspeed",dest = "windspeed", help="Wind Speed column name", required=True)
    parser.add_argument("-b", "--isbuffered", dest="isbuffered", help="Centerpath : 0 / buffered : 1", type=int, default=0)
    parser.add_argument("-r", "--bufferedradius", dest="bufferedradius", help="buffered radius", default=0, type=int)
    parser.add_argument('-d', "--debug", dest="debug", help="Display the dataframes and plots", default=False)

    args = parser.parse_args()
    if evaluate_input(args):
        shape_object = Shape_Generator(args.filename, args.name, args.year, args.windspeed,args.debug)
        if (args.isbuffered):
            shape_object.generate_buffered_file(args.bufferedradius)
        else:
            shape_object.generate_center_path()