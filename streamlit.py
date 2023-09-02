import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objs as go
import plotly.express as px 

import json 
import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')
    
forecast_description = '''
<b style='font-size:18px;'>
Forecasting revenue in Dombivli city for the financial year 2023-2024 on quarterly bases
</b>
'''

if 'redirect' not in st.session_state:
	st.session_state.redirect = True
if 'page_name' not in st.session_state:
	st.session_state.page_name = ''
if 'inventory' not in st.session_state:
	st.session_state.inventory = []  
if 'rev_page' not in st.session_state:
	st.session_state.rev_page = 0
if 'rev_name' not in st.session_state:
	st.session_state.rev_name = 'std'
if 'loc_name' not in st.session_state:
	st.session_state.loc_name = None
if 'cust_page' not in st.session_state:
	st.session_state.cust_page = 0

# contains forecasting csv
raw_data = pd.read_csv('Forecast_property.csv')

# contains forecasting models 
forecast = {}
filemap = json.load(open(r'forecast models/fileMap.json','rb'))

# here name is in lowercase 
for name in filemap.keys(): 
	forecast[name] = pickle.load(open(filemap[name],'rb'))

## ---------------------------------------------------------------------------------
pivot = pd.pivot_table(raw_data[['label','name']],index=['label','name']).copy(deep=True)
perm_com = {'LOCALITY':list(pivot.index.get_level_values(1)
							[pivot.index.get_level_values(0)=='locality'.upper()]),
			'PROJECT':list(pivot.index.get_level_values(1)
				  [pivot.index.get_level_values(0)=='project'.upper()])}

qDecode = pd.DataFrame(pd.to_datetime(raw_data.date).dropna().dt.to_period('Q').drop_duplicates())
qDecode = qDecode.set_index('date').sort_index().resample('Q').interpolate('nearest').index
qDecode = dict(enumerate([str(i.year)+'Q'+str(i.quarter) for i in qDecode]))
qDecode[64] = '2023Q1'
qDecode[65] = '2023Q2'
qDecode[66] = '2023Q3'
qDecode[67] = '2023Q4'
qDecode[68] = '2024Q1'
qEncode = dict(zip(list(qDecode.values()),list(qDecode.keys())))


def getData(label='locality', name='Dombivli East',method='nearest'):
	hold_data = raw_data[(raw_data.label==label.upper())&(raw_data.name==name)]
	hold_data.date = pd.to_datetime(hold_data.date)
	hold_data.dropna(inplace=True)
	hold_data = hold_data.drop(columns=['budgetRange','name','label','quarter'])
	hold_data.rename(columns={'price(per.sqft)':'price'},inplace=True)
	hold_data['Quarter'] = hold_data.date.dt.to_period('Q')
	hold_data.drop(columns=['date'],axis=1,inplace=True)
	hold_data.drop_duplicates(inplace=True)
	hold_data = hold_data.groupby(by='Quarter').mean()
	hold_data = hold_data.sort_index()
	hold_data = hold_data.resample('Q').interpolate(method)
	do_forecast = bool(sum([(i.year in [2022,2023]) for i in hold_data.index[-4:]]))
	return {'data':hold_data,'do_forecast':do_forecast}
		

def getKey(name):
	key = ''
	for i in perm_com.keys():
		if name in perm_com[i]:
			key = i
			return key 

def getSuper(x):
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
	res = x.maketrans(''.join(normal), ''.join(super_s))
	return x.translate(res)

def getName(x):
	return x.replace(getSuper('locality'),'').replace(getSuper('project'),'')

def revenueProfile(inv):
	# area,bed_room,floor,unit
	# get rates from forecast 
	unq_name = ([i[0] for i in inv])
	hold = []
	for name in set(unq_name):
		if name == 'Pendse Nagarˡᵒᶜᵃˡᶦᵗʸ':
			res = forecast[getName(name)].forecast(5).iloc[1:]
		else :
			res = forecast[getName(name)].forecast(4)
		x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in res.index]
		total = 0
		emp = np.zeros(len(res))
		for item in inv:
			if name == item[0]:
				# loop through forecast
				x = []
				for val in res.values.ravel():
					# apply model here 
					eps_carp = 8638.03568957
					eps_floor = -3228.20360704
					eps_br = 42580.48262206
					eps_foCa = 565.27354896
					est_revenue = item[-1]*(eps_carp*item[1]+eps_floor*item[3]+eps_br*item[2]+eps_foCa*val)
					x.append(est_revenue)
				emp += np.array(x)
		hold.append([name, x_axis, list(emp)])
	return hold


def update_default():
	st.session_state.loc_name = st.session_state.loc_n_values
##------------------------------------------------------------------
## home PAGE
if st.session_state.redirect:
	with st.form('Forecasting'):
		st.title('Forecast Property Rate') 
		st.markdown(forecast_description,unsafe_allow_html=True)
		st.image(r'f_cast.jpg')
		if st.form_submit_button('**Visit**'):
			st.session_state.redirect = False
			st.session_state.page_name = 'Forecast'
			st.experimental_rerun()

##------------------------------------------------------------------
## FORECASTING PAGE

if st.session_state.page_name == 'Forecast':
	# forecasting page 
	if st.sidebar.button('**Home Page**'):
		st.session_state.redirect = True
		st.session_state.page_name = ''
		st.experimental_rerun()


	rate_revenue = option_menu(menu_title=None,
							   options=['Price Rate','Revenue'],
							   icons=['bi-bar-chart','bi-cash-coin'],
						   	   orientation='horizontal')

	if rate_revenue == 'Price Rate':
		loc = st.sidebar.checkbox('LOCALITY',value=True)
		pro = st.sidebar.checkbox('PROJECT')
		value = []
		if loc :
			value += list([i+getSuper('locality') for i in perm_com['locality'.upper()]])
		if pro :
			value += list([i+getSuper('project') for i in perm_com['project'.upper()]])

		nameSelect = st.sidebar.selectbox('Name',tuple(value))
		if nameSelect :
			name = getName(nameSelect)
		else :
			name = ''
		do_forecast = st.sidebar.checkbox('Forecast Quartely Price', value=False)

		compare = st.sidebar.checkbox('Compare Properties')
		if compare:
			compareState = False
		else :
			compareState = True

		if name:
			st.title(nameSelect)
			response = getData(label=getKey(name),name=name)
			compWith = st.multiselect('compare with',tuple([x for x in value if x != nameSelect]),disabled=compareState)
			fig = go.Figure()
			emp = []
			x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in response['data'].index]
			fig.add_trace(go.Scatter(x=x_axis,
									 y=response['data'].values.ravel(),
									 mode='lines',
									 name=nameSelect))
			#  Do forecasting here 
			if do_forecast :
				pred = forecast[name].forecast(4)
				forecast_ax = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in pred.index]
				forecast_ax.insert(0,qEncode[str(response['data'].index[-1].year)+'Q'+str(response['data'].index[-1].quarter)])
				ls = list(pred.values.ravel())
				ls.insert(0,response['data'].values.ravel()[-1])
				fig.add_trace(go.Scatter(x=forecast_ax,
										 y=ls,
										 mode='lines+markers',
										 name='forecast_'+nameSelect,
										 line=dict(color='red')))
				x_axis += forecast_ax
			emp += x_axis
			if compWith:
				for compName in compWith:
					response = getData(label=getKey(getName(compName)),name=getName(compName))
					x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in response['data'].index]
					emp += x_axis
					fig.add_trace(go.Scatter(x=x_axis,
											 y=response['data'].values.ravel(),
											 mode='lines',
											 name=compName))
					if do_forecast:
						pred = forecast[getName(compName)].forecast(4)
						forecast_ax = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in pred.index]
						forecast_ax.insert(0,qEncode[str(response['data'].index[-1].year)+'Q'+str(response['data'].index[-1].quarter)])
						ls = list(pred.values.ravel())
						ls.insert(0,response['data'].values.ravel()[-1])
						fig.add_trace(go.Scatter(x=forecast_ax,
												 y=ls,
												 mode='lines+markers',
												 name='forecast_'+compName,
												 line=dict(color='red')))
						x_axis += forecast_ax

			fig.update_traces(marker=dict(size=4))
			fig.update_layout(
					xaxis=dict(
					tickmode='array',
					tickvals=list(set(emp)),
					ticktext=[qDecode[i] for i in set(emp)]
				),
					width=int(800),
					height=int(500),
					margin=dict(
					l=0
					)
			) 
			st.plotly_chart(fig)

	elif rate_revenue == 'Revenue':
		# add revenue prediction model here 
		with st.sidebar.form('add_invent'):
			# names
			value = []
			value += list([i+getSuper('locality') for i in perm_com['locality'.upper()]])
			value += list([i+getSuper('project') for i in perm_com['project'.upper()]])
			name = st.selectbox('Name',options=value)
			# area 
			area = st.number_input('Area',min_value=100.0,max_value=10000.0,step=1.0,value=500.0)
			# bedroom
			bedroom = st.number_input('Bed Room',min_value=1,max_value=4,step=1,value=1)
			# floor 
			floor = st.number_input('Floor',min_value=1,max_value=40,step=1,value=1)
			# units
			units = st.number_input('Units',min_value=1,max_value=100,step=1,value=1)
			if st.form_submit_button('Add'):
				st.session_state.inventory.append([name,area,bedroom,floor,units])
		
		with st.expander('**Inventory**'):
			st.info('Add Inventory')
			if st.session_state.inventory:
				count = 1
				state = []
				for name,area, bedRoom, floor, qty in st.session_state.inventory:
					with st.form('add_invent'+str(count)):
						col1, col2, col3, col4,col5 = st.columns(5)
						with col1:
							st.subheader(name)
						with col2 :
							st.metric(label='**Carpet Area**',value=area)
						with col3:
							st.metric(label='**Bed Room**',value=bedRoom)
						with col4:
							st.metric(label='**Floor**',value=floor)
						with col5:
							st.metric(label='**Unit**',value=qty)
						remove = st.form_submit_button('**Remove**')
						state.append(int(remove))
						count += 1
				if 1 in state:
					st.write(state.index(1))
					# remove element from list 
					st.session_state.inventory.pop(state.index(1))


					if st.session_state.rev_name == 'std':
						st.session_state.rev_page = 0
						# com -> con
					elif st.session_state.rev_name == 'con':
						st.session_state.rev_page = 1

					st.experimental_rerun()

		# Here add graphs
		with st.sidebar:
			res = option_menu(menu_title=None,
							  options=['Standalone','Consolidated'],
							  orientation='vertical',
							  default_index=st.session_state.rev_page)

		if st.session_state.inventory:
			# standalone OR consolidated 
			if res == 'Standalone':
				st.session_state.rev_name = 'std'
				# for each name(locality/project)
				# model(carpet_area,bedroom,floors,unit) -> total_price
				# plot each total_price
				fig = go.Figure()
				emp = []
				for _name,_ax,_est  in revenueProfile(st.session_state.inventory):
					# draw plot here 
					fig.add_trace(go.Scatter(x=_ax,
											 y=_est,
											 name=_name))
					emp += _ax
				fig.update_traces(marker=dict(size=4))
				fig.update_layout(
					xaxis=dict(
							tickmode='array',
							tickvals=list(set(emp)),
							ticktext=[qDecode[i] for i in set(emp)]
						),
					width=int(700),
					height=int(500),
					margin=dict(l=0),
					title=dict(
							text="<b style='font-size:25px'>Standalone Value of Inventory</b>"
						)
					)
				st.plotly_chart(fig)

			elif res == 'Consolidated':
				st.session_state.rev_name = 'con'
				# model(carpet_area,bedroom,floors,unit) -> total_price
				# plot sum of all total_prices
				rev_res = revenueProfile(st.session_state.inventory)
				axis = []
				for p in rev_res:
					axis += p[1]
				cons_ax = []
				cons_est = []
				nums = sorted(set(axis))
				for x_p in nums:
					i = 0
					cons_ax.append(x_p)
					for _ , new_ax, _val in rev_res:
						zp = dict(zip(new_ax,_val))
						if x_p in zp.keys():
							i += zp[x_p]
					cons_est.append(i)

				fig = go.Figure()
				fig.add_trace(go.Scatter(x=cons_ax,
										 y=cons_est,
										 name='Consolidated Value'))
				fig.update_traces(marker=dict(size=4))
				fig.update_layout(
					xaxis=dict(
							tickmode='array',
							tickvals=list(set(cons_ax)),
							ticktext=[qDecode[i] for i in set(cons_ax)]
						),
					width=int(700),
					height=int(500),
					margin=dict(l=0),
					title=dict(
							text="<b style='font-size:25px'>Consolidated Value of Inventory</b>",
						)
				)
				st.plotly_chart(fig)
				