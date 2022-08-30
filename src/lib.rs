#![feature(trait_alias)]
#![feature(cursor_remaining)]

use geo_types::geometry::{
    Coordinate, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use geo_types::Geometry;
use num_derive::FromPrimitive;
use num_traits::Float;
use num_traits::FromPrimitive;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;
use std::io::Read;

pub trait NumTy = Float + Debug + From<f64>;

pub enum WkbError {
    IoError(std::io::Error),
    Unsupported(&'static str),
    UnsupportedEndianess,
    InconsistentType,
}

impl From<std::io::Error> for WkbError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

#[derive(FromPrimitive)]
enum Endian {
    Big = 0,
    Little = 1,
}

#[derive(FromPrimitive)]
enum GeomTy {
    Point = 1,
    LineString = 2,
    Polygon = 3,
    MultiPoint = 4,
    MultiLineString = 5,
    MultiPolygon = 6,
    GeometryCollection = 7,
}

struct LittleEndianReader<'a, 'b> {
    cursor: &'b mut Cursor<&'a [u8]>,
}

impl<'a, 'b> LittleEndianReader<'a, 'b> {
    #[inline(always)]
    fn read_u32(&mut self) -> Result<u32, std::io::Error> {
        let mut r = [0u8; 4];
        self.cursor.read(&mut r)?;

        Ok(u32::from_le_bytes(r))
    }

    #[inline(always)]
    fn read_f64(&mut self) -> Result<f64, std::io::Error> {
        let mut r = [0u8; 8];
        self.cursor.read(&mut r)?;

        Ok(f64::from_le_bytes(r))
    }
}

pub struct Wkb(Vec<u8>);

impl Wkb {
    pub fn new(wkb: Vec<u8>) -> Self {
        Self(wkb)
    }
}

impl<T: NumTy> TryFrom<Geometry<T>> for Wkb {
    type Error = WkbError;

    fn try_from(_geom: Geometry<T>) -> Result<Wkb, Self::Error> {
        todo!();
    }
}

fn read_coordinate<T: NumTy>(reader: &mut LittleEndianReader) -> Result<Coordinate<T>, WkbError> {
    let x: T = reader.read_f64()?.into();
    let y: T = reader.read_f64()?.into();

    Ok(Coordinate { x, y })
}

fn read_coordinates<T: NumTy>(
    reader: &mut LittleEndianReader,
) -> Result<Vec<Coordinate<T>>, WkbError> {
    let num_points = reader.read_u32()? as usize;
    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        points.push(read_coordinate(reader)?);
    }

    Ok(points)
}

fn read_wkb<T: NumTy>(cursor: &mut Cursor<&[u8]>) -> Result<Geometry<T>, WkbError> {
    let mut endianness = [0u8; 1];
    cursor.read(&mut endianness)?;

    let endianness = Endian::from_u8(endianness[0])
        .ok_or_else(|| WkbError::Unsupported("unsupported endianness value"))?;
    let mut reader = match endianness {
        Endian::Big => return Err(WkbError::UnsupportedEndianess),
        Endian::Little => LittleEndianReader { cursor },
    };

    let ty = GeomTy::from_u32(reader.read_u32()?)
        .ok_or_else(|| WkbError::Unsupported("unsupported type value"))?;
    match ty {
        GeomTy::Point => read_coordinate(&mut reader).map(|p| Geometry::Point(Point::from(p))),
        GeomTy::LineString => {
            read_coordinates(&mut reader).map(|p| Geometry::LineString(LineString::from(p)))
        }
        GeomTy::Polygon => {
            let num_rings = reader.read_u32()? as usize;
            let mut rings = (0..num_rings)
                .map(|_| Ok(LineString::new(read_coordinates::<T>(&mut reader)?)))
                .collect::<Result<Vec<_>, WkbError>>()?;
            let exterior = rings.swap_remove(0);

            Ok(Geometry::Polygon(Polygon::new(exterior, rings)))
        }
        GeomTy::MultiPoint => {
            let num_points = reader.read_u32()? as usize;

            let points = (0..num_points)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::Point(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiPoint(MultiPoint::new(points)))
        }
        GeomTy::MultiLineString => {
            let num_linestrings = reader.read_u32()? as usize;

            let linestrings = (0..num_linestrings)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::LineString(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiLineString(MultiLineString::new(linestrings)))
        }
        GeomTy::MultiPolygon => {
            let num_polygons = reader.read_u32()? as usize;

            let polygons = (0..num_polygons)
                .map(|_| match read_wkb::<T>(cursor)? {
                    Geometry::Polygon(l) => Ok(l),
                    _ => Err(WkbError::InconsistentType),
                })
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::MultiPolygon(MultiPolygon::new(polygons)))
        }
        GeomTy::GeometryCollection => {
            let num_geometries = reader.read_u32()? as usize;
            let geometries = (0..num_geometries)
                .map(|_| read_wkb::<T>(cursor))
                .collect::<Result<Vec<_>, WkbError>>()?;
            Ok(Geometry::GeometryCollection(GeometryCollection::new_from(
                geometries,
            )))
        }
    }
}

impl<T: NumTy> TryFrom<Wkb> for Geometry<T> {
    type Error = WkbError;

    fn try_from(geom: Wkb) -> Result<Geometry<T>, Self::Error> {
        let mut cursor = Cursor::new(geom.0.as_slice());
        read_wkb(&mut cursor)
    }
}
