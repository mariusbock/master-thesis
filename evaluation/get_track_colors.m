function track_colors = get_track_colors(num_tracks, rnd_seed)

  track_colors2 = hsv(3*num_tracks);

  rand('twister', rnd_seed);
  rp = randperm(size(track_colors2, 1));

  for idx = 1:num_tracks
    track_colors(idx, :) = track_colors2(rp(idx), :);
  end

  disp(track_colors);


function track_colors = get_track_colors_old(num_tracks, rnd_seed)

  N = ceil(num_tracks/2);

  track_colors2 = hsv(N);
  track_colors2 = repmat(track_colors2, 2, 1);

  rand('twister', rnd_seed);

  %rand('twister', 64242);
  %rand('twister', 64247);

  rp = randperm(size(track_colors2, 1));

  track_colors = zeros(size(track_colors2, 1), 3);
  for i = 1:size(track_colors2, 1)
    track_colors(i, :) = track_colors2(rp(i), :);
  end
