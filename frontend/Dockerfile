# AlertIQ — Frontend dashboard image
#
# Multi-stage build: compile the Vite/React app, then serve the static
# build with a lightweight nginx image.
#
# Build:
#   docker build -t alertiq-frontend ./frontend
# Run:
#   docker run -p 5173:80 alertiq-frontend
# Or use docker-compose from the project root (recommended).

FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
