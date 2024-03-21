<script setup>
defineProps({
  station: Object,
});
</script>

<template>
  <div class="w-80 relative bg-slate-100 p-5 rounded shadow-md mb-4 hover:bg-slate-300">
    <h2 class="text-xl font-bold">{{ station.name }}</h2>
    <div class="flex my-2 text-gray-600 gap-2">
      <p class="font-semibold">Bike Stands: {{ station.bike_stands }}</p>
      <p class="font-semibold"
        :class="{ 'text-green-500': station.status === 'OPEN', 'text-red-500': station.status !== 'OPEN' }">
        Status: {{
      station.status }}</p>
    </div>
  </div>
</template>