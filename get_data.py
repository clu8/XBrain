import synapseclient

import config


syn = synapseclient.Synapse()

username = input('Username: ')
password = input('Password: ')

syn.login(username, password)

# Obtain a pointer and download the data
syn6174183 = syn.get('syn6174183', downloadLocation=config.DATA_PATH)

# Get the path to the local copy of the data file
filepath = syn6174183.path
