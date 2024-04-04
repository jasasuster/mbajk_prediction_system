import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

function StationInfo() {
  const { stationId } = useParams();
  const stationNumber = parseInt(stationId);
  const [station, setStation] = useState(null);

  const [predictions, setPredictions] = useState([]);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [error, setError] = useState(null);

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
      .then((data) => {
        const foundStation = data.find((s) => s.number === stationNumber);
        setStation(foundStation);
      });
  }, [stationNumber]);

  useEffect(() => {
    if (station) {
      const storedPredictions = getPredictions(station.number);
      if (storedPredictions) {
        setPredictions(storedPredictions);
        setIsLoadingPredictions(false);
      } else {
        setIsLoadingPredictions(true);
        fetch(`http://localhost:3000/mbajk/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ station_number: station.number }),
        })
          .then((res) => res.json())
          .then((data) => {
            storePredictions(station.number, data.predictions);
            setPredictions(data.predictions);
            setIsLoadingPredictions(false);
          })
          .catch((err) => {
            console.error('Error fetching predictions', err);
            setError(err.message);
            setIsLoadingPredictions(false);
          });
      }
    }
  }, [station]);

  function generateTimesArray(interval) {
    const result = [];
    const now = new Date();
    // Round up to the next full hour
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours() + 1, 0, 0);
    const end = new Date(start);
    end.setHours(start.getHours() + 7);

    for (let d = start; d <= end; d.setMinutes(d.getMinutes() + interval)) {
      result.push(format(d));
    }

    return result;
  }

  function format(inputDate) {
    let hours = inputDate.getHours();
    let minutes = inputDate.getMinutes();
    const formattedHours = hours === 0 ? 12 : hours < 10 ? "0" + hours : hours;
    const formattedMinutes = minutes < 10 ? "0" + minutes : minutes;
    return formattedHours + ":" + formattedMinutes;
  }

  const times = generateTimesArray(60);

  function storePredictions(stationNumber, predictions) {
    const now = new Date();
    const stationData = {
      number: stationNumber,
      predictions: predictions,
      timestamp: now.getTime(),
    };

    const jsonString = JSON.stringify(stationData);

    localStorage.setItem(`station-${stationNumber}`, jsonString);
  }

  function getPredictions(stationNumber) {
    const storedData = localStorage.getItem(`station-${stationNumber}`);
    if (storedData) {
      const stationData = JSON.parse(storedData);
      const now = new Date().getTime();
      // Check if the stored data is less than 1 hour old
      if (now - stationData.timestamp < 60 * 60 * 1000) {
        return stationData.predictions; 
      }
    }
    return null;
  }

  if (!station) return <div>Loading...</div>;

  return (
    <div className='bg-white max-w-xl mx-auto p-4 rounded-lg shadow-md'>
      <h2 className='text-2xl font-bold mb-2'>{station.name}</h2>
      <div className='flex items-center gap-2 font-medium mb-4'>
        <span className='text-gray-600'>Status: </span>
        <p className={`${station.status === 'OPEN' ? 'text-green-500' : 'text-red-500'}`}>{station.status === 'OPEN' ? 'Open Now' : 'Closed Now'}</p>
      </div>
      <div className='grid grid-cols-2 gap-4'>
        <p className='text-gray-600'>Bike Stands:</p>
        <p>{station.bike_stands}</p>
        <p className='text-gray-600'>Available Bike Stands:</p>
        <p>{station.available_bike_stands}</p>
        <p className='text-gray-600'>Available Bikes:</p>
        <p>{station.available_bikes}</p>
      </div>
      {isLoadingPredictions ? (
        <div>Loading predictions...</div>
      ) : (
        <div className='mt-4'>
          <h3 className='text-xl font-bold mb-2'>Predictions:</h3>
          {error ? (
            <div className='mt-4'>
              <p className='text-red-500'>Error: {error}</p>
            </div>
          ) : (
            <ul>
              {predictions.map((prediction, index) => (
                <li key={index} className='text-gray-600'>
                  {times[index]}: {prediction < 0 ? 0 : prediction > station.bike_stands ? station.bike_stands : prediction}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export default StationInfo;
