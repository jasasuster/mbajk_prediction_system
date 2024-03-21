<script setup>
import { ref } from 'vue';
import "leaflet/dist/leaflet.css";
import { LMap, LTileLayer, LMarker } from "@vue-leaflet/vue-leaflet";

const zoom = ref(13);
</script>

<template>
  <div class="w-[400px] h-full">
    <l-map ref="map" :zoom="zoom" :center="[46.55904166043249, 15.63807183545153]" :use-global-leaflet="false">
      <l-tile-layer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" name="OpenStreetMap"></l-tile-layer>
      <l-marker v-for="station in bikeStations" :key="station.number"
        :lat-lng="[station.position.lat, station.position.lng]">
      </l-marker>
    </l-map>
  </div>
</template>
