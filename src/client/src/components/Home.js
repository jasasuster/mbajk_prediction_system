import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import StationInfoCard from './StationInfoCard';

function Home() {
  const [stations, setStations] = useState([]);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    fetch('https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b', {
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    })
      .then((res) => {
        return res.json();
      })
      .then((data) => setStations(data));
  }, []);

  const filteredStations = stations.filter((station) => station.name.toLowerCase().includes(filter.toLowerCase()));

  const noMatchingStations = filter !== '' && filteredStations.length === 0;

  return (
    <div className='flex items-center justify-center'>
      <div className='bg-white p-8 rounded shadow-md w-[50vh]'>
        <h2 className='text-2xl font-bold mb-4'>MBajk Station overview</h2>
        <input
          type='text'
          placeholder='Filter...'
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className='border p-2 mb-4 rounded w-full'
        ></input>
        {noMatchingStations && <div className='text-red-500 mb-4 font-semibold text-center'>No gyms match your filter.</div>}
        <div className='h-[75vh] overflow-auto'>
          {filteredStations.map((station) => {
            return (
              <Link to={`/station/${station.number}`} key={station.number}>
                <StationInfoCard id={station.number} name={station.name} location={station.location} status={station.status} />
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default Home;
