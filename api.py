import requests
import socketio

base_url = 'https://catracker-api.fly.dev/api/v1/garages/'

sio = socketio.Client()
sio.connect('https://catracker-api.fly.dev')

class Garage:
    def __init__(self, garage):
        self.id = garage['_id']
        self.name = garage['garageName']
        self.address = garage['address']
        self.capacity = garage['capacity']
        self.cars_in_lot = garage['carsInLot']

    def to_dict(self):
        return {
            '_id': self.id,
            'carsInLot': int(self.cars_in_lot)
        }

def put_garage(garage: Garage):
    sio.emit('garageUpdate', garage.to_dict())

def get_garages():
    resp = requests.get(base_url)
    garages = []
    for g in resp.json():
        garages.append(Garage(g))
    return garages

