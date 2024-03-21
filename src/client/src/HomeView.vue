<script setup>
import { ref, onMounted } from 'vue';
import Map from './components/MapComponent.vue';
import StationCard from './components/StationCard.vue';

const bikeStations = ref([]);

onMounted(async () => {
  try {
    const response = await fetch('./src/assets/stations.json');
    const data = await response.json();
    bikeStations.value = data;
  } catch (error) {
    console.error('Error fetching stations:', error);
  }
});
</script>

<template>
  <div class="flex p-5 justify-center">
    <div class="mr-10 overflow-auto max-h-[80vh]">
      <StationCard v-for="station in bikeStations" :key="station.number" :station="station" />
    </div>
    <div class="w-1/2 max-h-[40vh]">
      <Map :bikeStations="bikeStations" />
    </div>
  </div>
</template>