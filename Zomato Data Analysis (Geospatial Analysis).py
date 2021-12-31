#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot


# In[2]:


df = pd.read_csv(r'C:\Users\HP\Downloads\3-Zomato Data Analysis-20211013T103556Z-001\3-Zomato Data Analysis/zomato.csv')
df.head()


# In[3]:


df.columns


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


# data prep processing for analysis


# In[7]:


df.isnull().sum()


# In[8]:


feature_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]
feature_na


# In[9]:


for feature in feature_na:
    print(' {} has {} missing values'.format(feature , np.round(df[feature].isnull().sum()/len(df)*100,4)))


# In[10]:


df['rate'].unique()


# In[11]:


df.dropna(axis='index',subset=['rate'],inplace=True)


# In[12]:


df.shape


# In[13]:


def split(x):
    return x.split('/')[0]


# In[14]:


df['rate'] = df['rate'].apply(split)


# In[15]:


df.head()


# In[16]:


df['rate'].unique()


# In[17]:


df.replace('NEW', 0 , inplace=True)


# In[18]:


df.replace('-', 0, inplace=True)


# In[19]:


df['rate'].dtype


# In[20]:


df['rate'] = df['rate'].astype(float)


# In[21]:


df.head()


# In[22]:


# in dept analysis of restaurant


# In[23]:


df_rate = df.groupby('name')['rate'].mean().to_frame().reset_index()
df_rate.columns = ['restaurant', 'avg_rating']
df_rate.head(20)


# In[24]:


sns.displot(df_rate['avg_rating'])


# In[25]:


# top restuarant chains in bengaluru


# In[26]:


chains = df['name'].value_counts()[0:20]
sns.barplot(x = chains, y = chains.index)
plt.title('Most famous restuarants in Bengaluru')
plt.xlabel('Number of outlets')


# In[27]:


x = df['online_order'].value_counts()
x


# In[28]:


labels = ['accepted', 'not accepted']
px.pie(df, values = x, labels = labels, title = 'Pie Chart')


# In[29]:


# analysing most famour restuarants


# In[30]:


x = df['book_table'].value_counts()
x


# In[31]:


labels = ['not book', 'book']
trace = go.Pie(labels=labels, values=x, hoverinfo = 'label+percent', textinfo = 'value')
iplot([trace])


# In[32]:


df['rest_type'].isna().sum()


# In[33]:


df['rest_type'].dropna(inplace=True)


# In[34]:


df['rest_type'].isna().sum()


# In[35]:


len(df['rest_type'].unique())


# In[36]:


trace1 = go.Bar(x = df['rest_type'].value_counts().nlargest(20).index, 
       y = df['rest_type'].value_counts().nlargest(20))
iplot([trace1])


# In[37]:


df.groupby('name')['votes'].sum().nlargest(20).plot.bar()


# In[38]:


trace2 = go.Bar(x = df.groupby('name')['votes'].sum().nlargest(20).index, 
                y = df.groupby('name')['votes'].sum().nlargest(20))
iplot([trace2])


# In[39]:


restaurant = []
location = []
for key,location_df in df.groupby('location'):
    location.append(key)
    restaurant.append(len(location_df['name'].unique()))


# In[40]:


df_total = pd.DataFrame(zip(location, restaurant))
df_total.head()


# In[41]:


df_total.columns = ['location', 'restaurant']
df_total.head()


# In[42]:


df_total.set_index('location', inplace=True)
df_total.head()


# In[43]:


df_total.sort_values(by='restaurant').tail(10).plot.bar()


# In[44]:


# analysing price of restaurant


# In[45]:


cuisines = df['cuisines'].value_counts()[0:10]
cuisines


# In[46]:


trace3 = go.Bar(x = df['cuisines'].value_counts()[0:10].index,
               y = df['cuisines'].value_counts()[0:10])


# In[47]:


iplot([trace3])


# In[48]:


df.columns


# In[49]:


df['approx_cost(for two people)'].isna().sum()


# In[50]:


df.dropna(axis='index', subset=['approx_cost(for two people)'], inplace=True)


# In[51]:


df['approx_cost(for two people)'].isna().sum()


# In[52]:


sns.displot(df['approx_cost(for two people)'])


# In[53]:


df['approx_cost(for two people)'].dtype


# In[54]:


df['approx_cost(for two people)'].unique()


# In[55]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',',''))


# In[56]:


df['approx_cost(for two people)'].unique()


# In[57]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(int)


# In[58]:


df['approx_cost(for two people)'].dtype


# In[59]:


sns.displot(df['approx_cost(for two people)'])


# In[60]:


sns.scatterplot(x = 'rate', y = 'approx_cost(for two people)', hue = 'online_order', data = df)


# In[61]:


sns.boxplot(x = 'online_order', y = 'votes', data = df)


# In[62]:


px.box(df, x = 'online_order', y = 'votes')


# In[63]:


px.box(df, x = 'online_order', y = 'approx_cost(for two people)')


# In[64]:


# analysing insights from Restuarant pattern


# In[65]:


df['approx_cost(for two people)'].min()


# In[66]:


df['approx_cost(for two people)'].max()


# In[67]:


df[df['approx_cost(for two people)'] == 6000]['name']


# In[68]:


data = df.copy()


# In[69]:


data.set_index('name', inplace=True)


# In[70]:


data.head()


# In[71]:


data['approx_cost(for two people)'].nlargest(10).plot.bar()


# In[72]:


data['approx_cost(for two people)'].nsmallest(10).plot.bar()


# In[73]:


data[data['approx_cost(for two people)'] <= 500]


# In[74]:


df_budget = data[data['approx_cost(for two people)'] <= 500].loc[:, ('approx_cost(for two people)')]
df_budget = df_budget.reset_index()
df_budget.head()


# In[75]:


#perform spatial analysis


# In[76]:


df[(df['rate']>4) & (df['approx_cost(for two people)'] <= 500)].shape


# In[77]:


len(df[(df['rate']>4) & (df['approx_cost(for two people)'] <= 500)]['name'].unique())


# In[78]:


df_new = df[(df['rate']>4) & (df['approx_cost(for two people)'] <= 500)]
df_new.head()


# In[79]:


location = []
total = []

for loc, location_df in df_new.groupby('location'):
    location.append(loc)
    total.append(len(location_df['name'].unique()))


# In[80]:


location_df = pd.DataFrame(zip(location,total))
location_df.head()


# In[81]:


location_df.columns = ['location', 'restaurant']
location_df.head()


# In[82]:


def return_budget(location, restaurant):
    budget = df[(df['approx_cost(for two people)']<=400)&(df['location']==location)&(df['rate']>4)&(df['rest_type']==restaurant)]
    return (budget['name'].unique())


# In[83]:


return_budget('BTM', 'Quick Bites')


# In[84]:


restaurant_location = df['location'].value_counts()[0:20]
sns.barplot(restaurant_location, restaurant_location.index)


# In[85]:


locations = pd.DataFrame({'Name':df['location'].unique()})
locations.head()


# In[86]:


from geopy.geocoders import Nominatim


# In[87]:


geolocator = Nominatim(user_agent = 'app')


# In[88]:


lat_lon = []
for location in locations['Name']:
    location = geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:
        geo = (location.latitude, location.longitude)
        lat_lon.append(geo)


# In[89]:


locations['geo_loc'] = lat_lon


# In[90]:


locations.head()


# In[91]:


locations.shape


# In[92]:


Rest_locations = pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.head()


# In[93]:


Rest_locations.columns = ['Name','count']
Rest_locations.head()


# In[94]:


Restaurant_locations = Rest_locations.merge(locations, on='Name', how='left').dropna()
Restaurant_locations.head()


# In[95]:


np.array(Restaurant_locations['geo_loc'])


# In[96]:


lat, lon = zip(*np.array(Restaurant_locations['geo_loc']))


# In[97]:


type(lat)


# In[98]:


Restaurant_locations['lat'] = lat
Restaurant_locations['lon'] = lon


# In[99]:


Restaurant_locations.head()


# In[100]:


Restaurant_locations.drop('geo_loc', axis=1, inplace=True)
Restaurant_locations.head()


# In[101]:


import folium
from folium.plugins import HeatMap


# In[102]:


def generatebasemap(default_location=[12.97, 77.59], default_zoom_start=12):
    basemap=folium.Map(location=default_location, zoom_start=default_zoom_start)
    return basemap


# In[103]:


basemap = generatebasemap()


# In[104]:


basemap


# In[105]:


HeatMap(Restaurant_locations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)


# In[106]:


basemap


# In[107]:


df.head()


# In[108]:


df2 = df[df['cuisines'] == 'North Indian']
df2.head()


# In[112]:


north_india = df2.groupby(['location'], as_index=False)['url'].agg('count')
north_india.head()


# In[114]:


north_india.columns = ['Name', 'count']
north_india.head()


# In[116]:


north_india = north_india.merge(locations, on='Name', how='left').dropna()
north_india.head()


# In[120]:


north_india['lat'],north_india['lon'] = zip(*north_india['geo_loc_x'].values)
north_india.head()


# In[119]:


north_india.drop('geo_loc',axis=1,inplace=True)
north_india.head()


# In[121]:


basemap=generatebasemap()
HeatMap(north_india[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap


# In[123]:


df_1 = df.groupby(['rest_type','name']).agg('count')
df_1


# In[128]:


dataset=df_1.sort_values(['url'], ascending=False).groupby(['rest_type'],as_index=False).apply(lambda x:x.sort_values(by='url',ascending=False))['url'].reset_index().rename(columns={'url':'count'})
dataset


# In[131]:


casual = dataset[dataset['rest_type']=='Casual Dining']
casual


# In[ ]:




