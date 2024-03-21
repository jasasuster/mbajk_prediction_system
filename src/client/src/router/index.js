import { createMemoryHistory, createRouter } from 'vue-router';

import HomeView from '../HomeView.vue';
import StationInfo from '../components/StationInfo.vue';

const routes = [
  { path: '/', component: HomeView },
  { path: '/stations/:id', component: StationInfo },
];

const router = createRouter({
  history: createMemoryHistory(),
  routes,
});

export default router;
