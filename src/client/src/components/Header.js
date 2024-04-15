import React from 'react';
import { Link } from 'react-router-dom';

export default function Header() {
  return (
    <nav className='flex items-center justify-between flex-wrap bg-red-500 p-6'>
      <div className='flex items-center flex-shrink-0 text-white mr-6'>
        <span className='font-semibold text-xl tracking-tight'>MBajk Stations</span>
      </div>
      <div className='w-full block flex-grow lg:flex lg:items-center lg:w-auto px-5'>
        <div className='text-sm lg:flex-grow'></div>
        <div>
          <div className='text-sm lg:flex-grow'>
            <Link to='/' className='block mt-4 lg:inline-block lg:mt-0 text-white hover:text-white mr-4'>
              Stations
            </Link>
            <Link to='/map' className='block mt-4 lg:inline-block lg:mt-0 text-white hover:text-white mr-4'>
              Map
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
