FROM node:20.10.0-alpine3.18

WORKDIR /app

# create new user called nonroot
RUN addgroup -S nonroot \
  && adduser -S nonroot -G nonroot

COPY . .

RUN npm install

ARG REACT_APP_SERVE_URL=p01--deploy-api--z7jwcpdkb6cf.code.run
ENV REACT_APP_SERVE_URL=$REACT_APP_SERVE_URL

RUN npm run build

ENV NODE_ENV=production

# switch to nonroot user
USER nonroot

EXPOSE 3001

CMD ["npx", "serve", "build", "-l", "3001"]