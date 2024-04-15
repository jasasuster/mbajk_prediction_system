const StationInfoCard = ({ number, name, location, status }) => {
  return (
    <div className='relative bg-slate-100 p-5 rounded shadow-md mb-4 hover:bg-slate-300'>
      <h2 className='text-xl font-bold'>{name}</h2>
      <div className='flex my-2 text-gray-600 gap-2'>
        <p className='font-semibold'>{location}</p>
      </div>
      <span className={`text-lg ${status === 'OPEN' ? 'text-green-500' : 'text-red-500'}`}>{status === 'OPEN' ? 'Open Now' : 'Closed Now'}</span>
    </div>
  );
};

export default StationInfoCard;
