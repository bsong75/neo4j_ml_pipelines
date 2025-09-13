

sql=sql_yaml.get_sql("line_features', {"filer":filer, "entry_nbr":entry_nbr, "line_nbr": str(line_nbr)})
cursor =db.get_cursor()
cursor.execute(sql)
row_line_features=cursor.fetchone()


#  dr_prediction server()

headers= { 
    'X-DataRobot-Model-Cache-Hit': 'true',
    'Content-Type': 'text/plain; charset=UTF-8',
    'Authorization': 'Bearer {}'.format(self.config_params['DR_API_TOKEN'])
    }
prediction_response = requests.post(
    f'{self.config_parms["DR_ENDPOINT"]}{model_deployment_id}/predictions',
    data=features,
    headers=headers,
    verify=False
)

response = predictions_response.json()
